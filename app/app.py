
from pathlib import Path
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
st.set_page_config(page_title="Energy Intelligence System", page_icon="⚡", layout="wide")
st.markdown("""
<style>
.stApp{background:#f5f7fb;color:#172033}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e5eaf2}
.block-container{padding-top:1.5rem;padding-bottom:2rem}
.hero{background:linear-gradient(115deg,#10243e,#176b72);padding:28px;border-radius:18px;color:white;margin-bottom:20px}
.hero h1{color:white;font-size:32px;margin:0}
.hero p{color:#d8e9ed;margin:8px 0 0}
.kpi{background:white;border:1px solid #e5eaf2;border-radius:14px;padding:17px}
.kpi-label{color:#667085;font-size:13px}
.kpi-value{font-size:25px;font-weight:750;color:#14253d;margin-top:5px}
.small-note{color:#667085;font-size:12px}
h1,h2,h3{color:#172b45}
</style>
""", unsafe_allow_html=True)

def find_files():
    files=[]
    for folder in [DATA_DIR]:
        if folder.exists():
            files += [p for p in folder.rglob("*") if p.suffix.lower() in [".csv",".xlsx",".xls"]]
    return sorted(set(files))

@st.cache_data(show_spinner=False)
def read_dataset(source, modified):
    path=Path(source)
    if path.suffix.lower()==".csv":
        return pd.read_csv(path,low_memory=False)
    return pd.read_excel(path)

def standardize(df):
    df=df.copy()
    df.columns=[str(c).strip() for c in df.columns]
    if "Datetime" not in df.columns:
        if "Date" in df.columns and "Time" in df.columns:
            df["Datetime"]=pd.to_datetime(df["Date"].astype(str)+" "+df["Time"].astype(str),dayfirst=True,errors="coerce")
        elif "Date" in df.columns:
            df["Datetime"]=pd.to_datetime(df["Date"],errors="coerce",dayfirst=True)
    if "Datetime" in df.columns:
        df["Datetime"]=pd.to_datetime(df["Datetime"],errors="coerce")
    if "Global_active_power" not in df.columns:
        candidates=[c for c in df.columns if "active_power" in c.lower() or c.lower() in ["power","consumption","globalactivepower"]]
        if candidates: df.rename(columns={candidates[0]:"Global_active_power"},inplace=True)
    if "Global_active_power" in df.columns:
        df["Global_active_power"]=pd.to_numeric(df["Global_active_power"],errors="coerce")
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    if "Datetime" in df.columns: df=df.dropna(subset=["Datetime"]).sort_values("Datetime")
    return df.reset_index(drop=True)

def load_data():
    files=find_files()
    uploaded=st.sidebar.file_uploader("Upload CSV or Excel",type=["csv","xlsx","xls"])
    if uploaded:
        try:
            raw=pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
            return standardize(raw),uploaded.name
        except Exception as e:
            st.sidebar.error(f"Upload error: {e}")
    if not files:
        return None,None
    choices={str(p.relative_to(ROOT)):p for p in files}
    selected=st.sidebar.selectbox("Select dataset",list(choices.keys()))
    p=choices[selected]
    try:
        return standardize(read_dataset(str(p),p.stat().st_mtime)),p.name
    except Exception as e:
        st.error(f"Cannot read dataset: {e}")
        return None,None

def metric_card(label,value,help_text=None):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',unsafe_allow_html=True)
    if help_text: st.caption(help_text)

def prepare_hourly(df):
    if "Datetime" not in df or "Global_active_power" not in df: return pd.DataFrame()
    x=df[["Datetime","Global_active_power"]].dropna().copy()
    x=x.set_index("Datetime").resample("h").mean()
    x["Global_active_power"]=x["Global_active_power"].interpolate(method="time",limit_direction="both")
    return x.dropna().reset_index()

def add_features(h):
    x=h.copy()
    dt=x["Datetime"]
    x["Hour"]=dt.dt.hour
    x["Day"]=dt.dt.day
    x["Month"]=dt.dt.month
    x["DayOfWeek"]=dt.dt.dayofweek
    x["IsWeekend"]=(x["DayOfWeek"]>=5).astype(int)
    y=x["Global_active_power"]
    x["Lag_1"]=y.shift(1)
    x["Lag_24"]=y.shift(24)
    x["Lag_168"]=y.shift(168)
    x["Rolling_Mean_24"]=y.shift(1).rolling(24).mean()
    x["Rolling_Mean_168"]=y.shift(1).rolling(168).mean()
    return x.dropna().reset_index(drop=True)

FEATURES=["Hour","Day","Month","DayOfWeek","IsWeekend","Lag_1","Lag_24","Lag_168","Rolling_Mean_24","Rolling_Mean_168"]
@st.cache_data(show_spinner=False)
def train_models(hourly):
    f=add_features(hourly)
    if len(f)<300: return None
    X=f[FEATURES]
    y=f["Global_active_power"]
    cut=int(len(f)*0.8)
    X_train,X_test=X.iloc[:cut],X.iloc[cut:]
    y_train,y_test=y.iloc[:cut],y.iloc[cut:]
    models={
        "Linear Regression":LinearRegression(),
        "Random Forest":RandomForestRegressor(n_estimators=120,max_depth=14,min_samples_leaf=2,n_jobs=-1,random_state=42),
        "Hist Gradient Boosting":HistGradientBoostingRegressor(max_iter=120,learning_rate=0.08,max_leaf_nodes=20,random_state=42)
    }
    results=[]; predictions={}
    for name,model in models.items():
        model.fit(X_train,y_train)
        pred=model.predict(X_test)
        results.append({"Model":name,"MAE":mean_absolute_error(y_test,pred),"RMSE":np.sqrt(mean_squared_error(y_test,pred)),"R2":r2_score(y_test,pred)})
        predictions[name]=pd.DataFrame({"Datetime":f["Datetime"].iloc[cut:].values,"Actual":y_test.values,"Predicted":pred})
    scores=pd.DataFrame(results).sort_values("RMSE")
    return f,models,scores,predictions

def future_forecast(hourly,model,hours=24):
    hist=hourly["Global_active_power"].astype(float).tolist()
    last_time=hourly["Datetime"].iloc[-1]
    rows=[]
    for i in range(1,hours+1):
        dt=last_time+pd.Timedelta(hours=i)
        if len(hist)<168: break
        row={"Hour":dt.hour,"Day":dt.day,"Month":dt.month,"DayOfWeek":dt.dayofweek,"IsWeekend":int(dt.dayofweek>=5),
             "Lag_1":hist[-1],"Lag_24":hist[-24],"Lag_168":hist[-168],
             "Rolling_Mean_24":np.mean(hist[-24:]),"Rolling_Mean_168":np.mean(hist[-168:])}
        pred=float(model.predict(pd.DataFrame([row])[FEATURES])[0])
        pred=max(0,pred)
        rows.append({"Datetime":dt,"Forecast":pred})
        hist.append(pred)
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def detect_anomalies(hourly,contamination):
    x=hourly.copy()
    if len(x)<100: return x.assign(Anomaly="Normal",Anomaly_Score=0)
    model=IsolationForest(contamination=contamination,n_estimators=150,random_state=42)
    features=x[["Global_active_power"]].copy()
    features["Hour"]=x["Datetime"].dt.hour
    features["DayOfWeek"]=x["Datetime"].dt.dayofweek
    labels=model.fit_predict(features)
    x["Anomaly"]=np.where(labels==-1,"Anomaly","Normal")
    x["Anomaly_Score"]=-model.decision_function(features)
    return x

def make_excel(dataframes):
    output=BytesIO()
    with pd.ExcelWriter(output,engine="openpyxl") as writer:
        for name,data in dataframes.items():
            data.to_excel(writer,sheet_name=name[:31],index=False)
    return output.getvalue()

def make_pdf(summary):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        output=BytesIO()
        doc=SimpleDocTemplate(output,pagesize=A4)
        styles=getSampleStyleSheet()
        story=[Paragraph("Energy Intelligence System - Report",styles["Title"]),Spacer(1,8*mm)]
        rows=[["Metric","Value"]]+[[str(k),str(v)] for k,v in summary.items()]
        table=Table(rows,colWidths=[75*mm,85*mm])
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#173b54")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.lightgrey),("PADDING",(0,0),(-1,-1),7)]))
        story.append(table)
        doc.build(story)
        return output.getvalue()
    except Exception:
        return None

def show_chart(fig):
    fig.update_layout(template="plotly_white",margin=dict(l=10,r=10,t=45,b=10),font=dict(color="#24364b"))
    st.plotly_chart(fig,use_container_width=True)

df,source_name=load_data()
st.markdown('<div class="hero"><h1>⚡ Energy Intelligence System</h1><p>Energy Analytics • Machine Learning • Forecasting • Sustainability</p></div>',unsafe_allow_html=True)
if df is None:
    st.warning("No dataset found. Put a CSV, XLSX or XLS file in data/ or data/processed, or upload one from the sidebar.")
    st.stop()
if "Datetime" not in df.columns or "Global_active_power" not in df.columns:
    st.error("The dataset must contain Datetime (or Date and Time) and Global_active_power columns.")
    st.stop()

df=df.dropna(subset=["Datetime","Global_active_power"]).copy()
df=df[df["Global_active_power"]>=0]
if df.empty:
    st.error("No valid consumption records were found.")
    st.stop()

min_date=df["Datetime"].min().date()
max_date=df["Datetime"].max().date()
st.sidebar.markdown("---")
st.sidebar.subheader("Analysis Filters")
date_range=st.sidebar.date_input("Date range",[min_date,max_date],min_value=min_date,max_value=max_date)
if isinstance(date_range,(tuple,list)) and len(date_range)==2:
    start=pd.Timestamp(date_range[0]); end=pd.Timestamp(date_range[1])+pd.Timedelta(days=1)
    view=df[(df["Datetime"]>=start)&(df["Datetime"]<end)].copy()
else:
    view=df.copy()
if view.empty:
    st.warning("No records in the selected date range.")
    st.stop()

hourly=prepare_hourly(view)
if hourly.empty:
    st.error("Unable to create hourly data.")
    st.stop()
page=st.sidebar.radio("Navigation",[
"Executive Overview","Consumption Analytics","Advanced Energy Analysis","Peak Demand Monitoring",
"AI Forecasting Studio","Model Comparison Lab","Anomaly Detection Center","Energy Cost Calculator",
"Carbon Emission Monitor","What-if Simulator","Automated Insights & Reports","Data Management & Explorer"
])

avg=float(view["Global_active_power"].mean())
peak=float(view["Global_active_power"].max())
energy=float(hourly["Global_active_power"].sum())
records=len(view)
if page=="Executive Overview":
    st.subheader("Executive Overview")
    cols=st.columns(5)
    for c,(label,val) in zip(cols,[("Average Power",f"{avg:.3f} kW"),("Peak Power",f"{peak:.3f} kW"),("Estimated Energy",f"{energy:,.1f} kWh"),("Records",f"{records:,}"),("Time Coverage",f"{(view['Datetime'].max()-view['Datetime'].min()).days} days")]):
        with c: metric_card(label,val)
    freq=st.selectbox("Trend aggregation",["Hourly","Daily","Weekly","Monthly"],index=1)
    rule={"Hourly":"h","Daily":"D","Weekly":"W","Monthly":"MS"}[freq]
    trend=hourly.set_index("Datetime")["Global_active_power"].resample(rule).mean().reset_index()
    trend["Rolling Average"]=trend["Global_active_power"].rolling(7,min_periods=1).mean()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=trend["Datetime"],y=trend["Global_active_power"],name="Consumption",mode="lines"))
    fig.add_trace(go.Scatter(x=trend["Datetime"],y=trend["Rolling Average"],name="Rolling average",mode="lines"))
    show_chart(fig)
    c1,c2=st.columns(2)
    with c1:
        profile=hourly.assign(Hour=hourly["Datetime"].dt.hour).groupby("Hour")["Global_active_power"].mean().reset_index()
        show_chart(px.bar(profile,x="Hour",y="Global_active_power",title="Average Consumption by Hour"))
    with c2:
        weekdays=hourly.assign(Weekday=hourly["Datetime"].dt.day_name()).groupby("Weekday")["Global_active_power"].mean().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).reset_index()
        show_chart(px.bar(weekdays,x="Weekday",y="Global_active_power",title="Average Consumption by Weekday"))
    st.caption(f"Selected data source: {source_name}. Energy is estimated from hourly average power.")

elif page=="Consumption Analytics":
    st.subheader("Consumption Analytics")
    c1,c2=st.columns(2)
    with c1:
        show_chart(px.histogram(view,x="Global_active_power",nbins=50,title="Power Consumption Distribution"))
    with c2:
        bymonth=hourly.assign(Month=hourly["Datetime"].dt.month_name()).groupby("Month")["Global_active_power"].mean().reset_index()
        show_chart(px.bar(bymonth,x="Month",y="Global_active_power",title="Average Consumption by Month"))
    hourly["Hour"]=hourly["Datetime"].dt.hour
    hourly["Month"]=hourly["Datetime"].dt.month
    pivot=hourly.pivot_table(index="Hour",columns="Month",values="Global_active_power",aggfunc="mean")
    show_chart(px.imshow(pivot,aspect="auto",title="Consumption Heatmap: Hour × Month",labels=dict(x="Month",y="Hour",color="Power")))
    if all(c in view.columns for c in ["Sub_metering_1","Sub_metering_2","Sub_metering_3"]):
        sub=view[["Sub_metering_1","Sub_metering_2","Sub_metering_3"]].apply(pd.to_numeric,errors="coerce").mean().reset_index()
        sub.columns=["Sub-metering","Average"]
        show_chart(px.bar(sub,x="Sub-metering",y="Average",title="Average Sub-metering Values"))
    st.dataframe(view.describe(include="all").T,use_container_width=True)

elif page=="Advanced Energy Analysis":
    st.subheader("Advanced Energy Analysis")
    c1,c2=st.columns(2)
    with c1:
        byday=hourly.assign(Date=hourly["Datetime"].dt.date).groupby("Date")["Global_active_power"].agg(["mean","max"]).reset_index()
        show_chart(px.line(byday,x="Date",y=["mean","max"],title="Daily Average and Maximum Power"))
    with c2:
        monthly=hourly.assign(YearMonth=hourly["Datetime"].dt.to_period("M").astype(str)).groupby("YearMonth")["Global_active_power"].mean().reset_index()
        show_chart(px.line(monthly,x="YearMonth",y="Global_active_power",markers=True,title="Monthly Consumption Trend"))
    st.markdown("#### Period comparison")
    monthly=hourly.set_index("Datetime")["Global_active_power"].resample("MS").mean()
    if len(monthly)>=2:
        current=float(monthly.iloc[-1]); previous=float(monthly.iloc[-2])
        change=(current-previous)/previous*100 if previous else 0
        st.metric("Latest month vs previous month",f"{change:+.2f}%",f"{current-previous:+.3f} kW")
    else: st.info("At least two months of data are needed for month-to-month comparison.")
    if "Voltage" in view.columns:
        c=view[["Voltage","Global_active_power"]].apply(pd.to_numeric,errors="coerce").dropna()
        if len(c)>10: show_chart(px.scatter(c.sample(min(5000,len(c)),random_state=42),x="Voltage",y="Global_active_power",title="Voltage vs Active Power"))

elif page=="Peak Demand Monitoring":
    st.subheader("Peak Demand Monitoring")
    hourly["Date"]=hourly["Datetime"].dt.date
    daily=hourly.groupby("Date")["Global_active_power"].agg(Average="mean",Peak="max").reset_index()
    c1,c2=st.columns(2)
    with c1: show_chart(px.line(daily,x="Date",y="Peak",title="Daily Peak Power"))
    with c2: show_chart(px.bar(daily.nlargest(15,"Peak"),x="Date",y="Peak",title="15 Highest Peak Days"))
    peak_rows=hourly.nlargest(20,"Global_active_power")[["Datetime","Global_active_power"]]
    st.markdown("#### Highest demand hours")
    st.dataframe(peak_rows,use_container_width=True)
    threshold=st.slider("Peak threshold (kW)",float(hourly["Global_active_power"].quantile(.5)),float(hourly["Global_active_power"].max()),float(hourly["Global_active_power"].quantile(.95)))
    high=hourly[hourly["Global_active_power"]>=threshold]
    st.metric("Hours above threshold",f"{len(high):,}")
    st.dataframe(high,use_container_width=True)

elif page in ["AI Forecasting Studio","Model Comparison Lab"]:
    st.subheader(page)
    if len(hourly)<400:
        st.warning("At least 400 hourly records are recommended for model training.")
    else:
        with st.spinner("Training forecasting models..."):
            trained=train_models(hourly)
        if trained is None:
            st.warning("Not enough data after feature engineering.")
        else:
            f,models,scores,predictions=trained
            best_name=scores.iloc[0]["Model"]
            if page=="Model Comparison Lab":
                st.markdown("#### Model performance (chronological 80/20 split)")
                st.dataframe(scores.style.format({"MAE":"{:.4f}","RMSE":"{:.4f}","R2":"{:.4f}"}),use_container_width=True)
                show_chart(px.bar(scores,x="Model",y="RMSE",title="Model Comparison - RMSE"))
                selected=st.selectbox("Inspect model",scores["Model"].tolist())
                pred=predictions[selected]
                c1,c2=st.columns(2)
                with c1: show_chart(px.line(pred,x="Datetime",y=["Actual","Predicted"],title=f"{selected}: Actual vs Predicted"))
                with c2: show_chart(px.scatter(pred,x="Actual",y="Predicted",title="Prediction Fit"))
                st.download_button("Download model comparison",make_excel({"Scores":scores,"Predictions":pred}),"model_comparison.xlsx")
            else:
                st.metric("Best model by RMSE",best_name,f"RMSE: {scores.iloc[0]['RMSE']:.4f}")
                horizon=st.select_slider("Forecast horizon",options=[24,48,72,168],value=24)
                future=future_forecast(hourly,models[best_name],horizon)
                if not future.empty:
                    recent=hourly.tail(24*7)
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=recent["Datetime"],y=recent["Global_active_power"],name="Historical"))
                    fig.add_trace(go.Scatter(x=future["Datetime"],y=future["Forecast"],name="Forecast",line=dict(dash="dash")))
                    show_chart(fig)
                    st.dataframe(future,use_container_width=True)
                    st.download_button("Download forecast CSV",future.to_csv(index=False).encode("utf-8"),"energy_forecast.csv")
                else: st.warning("Not enough history for the selected forecast.")

elif page=="Anomaly Detection Center":
    st.subheader("Anomaly Detection Center")
    contamination=st.slider("Expected anomaly proportion",0.005,0.10,0.02,0.005)
    with st.spinner("Detecting unusual consumption patterns..."):
        anomalies=detect_anomalies(hourly,contamination)
    count=int((anomalies["Anomaly"]=="Anomaly").sum())
    c1,c2,c3=st.columns(3)
    with c1: metric_card("Detected anomalies",f"{count:,}")
    with c2: metric_card("Anomaly rate",f"{count/len(anomalies)*100:.2f}%")
    with c3: metric_card("Detection method","Isolation Forest")
    show_chart(px.scatter(anomalies,x="Datetime",y="Global_active_power",color="Anomaly",title="Anomaly Timeline"))
    unusual=anomalies[anomalies["Anomaly"]=="Anomaly"].sort_values("Anomaly_Score",ascending=False)
    st.dataframe(unusual,use_container_width=True)
    st.download_button("Download anomaly results",anomalies.to_csv(index=False).encode("utf-8"),"anomaly_results.csv")

elif page=="Energy Cost Calculator":
    st.subheader("Energy Cost Calculator")
    tariff=st.number_input("Electricity price per kWh",min_value=0.0,value=0.15,step=0.01)
    currency=st.text_input("Currency label","EUR")
    total_kwh=float(hourly["Global_active_power"].sum())
    cost=total_kwh*tariff
    c1,c2,c3=st.columns(3)
    with c1: metric_card("Estimated energy",f"{total_kwh:,.2f} kWh")
    with c2: metric_card("Estimated cost",f"{cost:,.2f} {currency}")
    with c3: metric_card("Average hourly cost",f"{cost/len(hourly):,.4f} {currency}")
    daily=hourly.assign(Date=hourly["Datetime"].dt.date).groupby("Date")["Global_active_power"].sum().reset_index()
    daily["Estimated Cost"]=daily["Global_active_power"]*tariff
    show_chart(px.line(daily,x="Date",y="Estimated Cost",title="Estimated Daily Electricity Cost"))
    st.info("This is an estimate based on hourly average active power and the tariff you enter. It does not include taxes, fixed charges or time-of-use pricing.")
    st.download_button("Download cost analysis",daily.to_csv(index=False).encode("utf-8"),"energy_cost_analysis.csv")

elif page=="Carbon Emission Monitor":
    st.subheader("Carbon Emission Monitor")
    factor=st.number_input("Emission factor (kg CO₂ per kWh)",min_value=0.0,value=0.4,step=0.05)
    total=float(hourly["Global_active_power"].sum())
    emissions=total*factor
    c1,c2=st.columns(2)
    with c1: metric_card("Estimated energy",f"{total:,.2f} kWh")
    with c2: metric_card("Estimated emissions",f"{emissions:,.2f} kg CO₂")
    daily=hourly.assign(Date=hourly["Datetime"].dt.date).groupby("Date")["Global_active_power"].sum().reset_index()
    daily["CO2 kg"]=daily["Global_active_power"]*factor
    show_chart(px.area(daily,x="Date",y="CO2 kg",title="Estimated Daily Carbon Emissions"))
    st.caption("Emission factor is user-defined. Results are estimates, not verified environmental measurements.")

elif page=="What-if Simulator":
    st.subheader("What-if Energy Savings Simulator")
    reduction=st.slider("Simulated consumption reduction",0,50,10,1)
    tariff=st.number_input("Electricity tariff per kWh",min_value=0.0,value=0.15,step=0.01)
    baseline=float(hourly["Global_active_power"].sum())
    saved=baseline*reduction/100
    c1,c2,c3=st.columns(3)
    with c1: metric_card("Current energy",f"{baseline:,.2f} kWh")
    with c2: metric_card("Potential energy saved",f"{saved:,.2f} kWh")
    with c3: metric_card("Potential cost saving",f"{saved*tariff:,.2f}")
    compare=pd.DataFrame({"Scenario":["Current","After reduction"],"Energy (kWh)":[baseline,baseline-saved]})
    show_chart(px.bar(compare,x="Scenario",y="Energy (kWh)",title="Consumption Scenario Comparison"))
    st.info("This is a scenario simulation, not a guaranteed saving. Actual savings depend on household behavior and equipment.")

elif page=="Automated Insights & Reports":
    st.subheader("Automated Insights & Reports")
    daily=hourly.assign(Date=hourly["Datetime"].dt.date).groupby("Date")["Global_active_power"].sum()
    peak_hour=int(hourly.groupby(hourly["Datetime"].dt.hour)["Global_active_power"].mean().idxmax())
    low_hour=int(hourly.groupby(hourly["Datetime"].dt.hour)["Global_active_power"].mean().idxmin())
    insights=[
        f"Average active power is {avg:.3f} kW.",
        f"Maximum observed power is {peak:.3f} kW.",
        f"Estimated energy in the selected period is {energy:,.2f} kWh.",
        f"Hour {peak_hour}:00 has the highest average hourly consumption.",
        f"Hour {low_hour}:00 has the lowest average hourly consumption.",
        f"Daily energy consumption averages {daily.mean():.2f} kWh."
    ]
    for item in insights: st.info(item)
    summary={"Source":source_name,"Records":records,"Average Power kW":round(avg,4),"Peak Power kW":round(peak,4),"Estimated Energy kWh":round(energy,2),"Peak Consumption Hour":peak_hour,"Lowest Consumption Hour":low_hour}
    c1,c2=st.columns(2)
    with c1:
        st.download_button("Download Excel report",make_excel({"Summary":pd.DataFrame([summary]),"Daily Energy":daily.reset_index()}),"energy_report.xlsx")
    with c2:
        pdf=make_pdf(summary)
        if pdf: st.download_button("Download PDF report",pdf,"energy_report.pdf")
        else: st.warning("Install reportlab to enable PDF export.")
    st.markdown("#### Daily consumption")
    show_chart(px.line(daily.reset_index(),x="Date",y="Global_active_power",title="Daily Energy Consumption"))

elif page=="Data Management & Explorer":
    st.subheader("Data Management & Explorer")
    c1,c2,c3=st.columns(3)
    with c1: metric_card("Rows",f"{len(view):,}")
    with c2: metric_card("Missing values",f"{int(view.isna().sum().sum()):,}")
    with c3: metric_card("Duplicate rows",f"{int(view.duplicated().sum()):,}")
    st.markdown("#### Data quality")
    quality=pd.DataFrame({"Column":view.columns,"Data Type":[str(x) for x in view.dtypes],"Missing":[int(view[c].isna().sum()) for c in view.columns],"Unique":[int(view[c].nunique()) for c in view.columns]})
    st.dataframe(quality,use_container_width=True)
    search=st.text_input("Search in displayed data")
    filtered=view
    if search:
        mask=filtered.astype(str).apply(lambda col:col.str.contains(search,case=False,na=False)).any(axis=1)
        filtered=filtered[mask]
    page_size=st.selectbox("Rows per page",[25,50,100,250],index=1)
    total_pages=max(1,int(np.ceil(len(filtered)/page_size)))
    page_num=st.number_input("Page",min_value=1,max_value=total_pages,value=1)
    st.caption(f"{len(filtered):,} matching rows • Page {page_num} of {total_pages}")
    st.dataframe(filtered.iloc[(page_num-1)*page_size:page_num*page_size],use_container_width=True)
    st.download_button("Download filtered CSV",filtered.to_csv(index=False).encode("utf-8"),"filtered_energy_data.csv")
    st.download_button("Download filtered Excel",make_excel({"Filtered Data":filtered}),"filtered_energy_data.xlsx")
