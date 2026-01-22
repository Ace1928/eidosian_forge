# IMPORTING PACKAGES

import pandas as pd
import requests
import pandas_ta as ta
import matplotlib.pyplot as plt
from termcolor import colored as cl
import math

plt.rcParams["figure.figsize"] = (20, 10)
plt.style.use("fivethirtyeight")


# EXTRACTING HISTORICAL DATA
def get_historical_data(symbol, start_date, interval):
    url = "https://api.benzinga.com/api/v2/bars"
    querystring = {
        "token": "YOUR API KEY",
        "symbols": f"{symbol}",
        "from": f"{start_date}",
        "interval": f"{interval}",
    }

    hist_json = requests.get(url, params=querystring).json()
    df = pd.DataFrame(hist_json[0]["candles"])

    return df


aapl = get_historical_data("AAPL", "1993-01-01", "1W")
aapl.tail()

# CALCULATING DONCHIAN CHANNEL

aapl[["dcl", "dcm", "dcu"]] = aapl.ta.donchian(lower_length=40, upper_length=50)
aapl = aapl.dropna().drop("time", axis=1).rename(columns={"dateTime": "date"})
aapl = aapl.set_index("date")
aapl.index = pd.to_datetime(aapl.index)

aapl.tail()

# PLOTTING DONCHIAN CHANNEL

plt.plot(aapl[-300:].close, label="CLOSE")
plt.plot(aapl[-300:].dcl, color="black", linestyle="--", alpha=0.3)
plt.plot(aapl[-300:].dcm, color="orange", label="DCM")
plt.plot(aapl[-300:].dcu, color="black", linestyle="--", alpha=0.3, label="DCU,DCL")
plt.legend()
plt.title("AAPL DONCHIAN CHANNELS 50")
plt.xlabel("Date")
plt.ylabel("Close")

# BACKTESTING THE STRATEGY


def implement_strategy(aapl, investment):

    in_position = False
    equity = investment

    for i in range(3, len(aapl)):
        if aapl["high"][i] == aapl["dcu"][i] and in_position == False:
            no_of_shares = math.floor(equity / aapl.close[i])
            equity -= no_of_shares * aapl.close[i]
            in_position = True
            print(
                cl("BUY: ", color="green", attrs=["bold"]),
                f"{no_of_shares} Shares are bought at ${aapl.close[i]} on {str(aapl.index[i])[:10]}",
            )
        elif aapl["low"][i] == aapl["dcl"][i] and in_position == True:
            equity += no_of_shares * aapl.close[i]
            in_position = False
            print(
                cl("SELL: ", color="red", attrs=["bold"]),
                f"{no_of_shares} Shares are bought at ${aapl.close[i]} on {str(aapl.index[i])[:10]}",
            )
    if in_position == True:
        equity += no_of_shares * aapl.close[i]
        print(
            cl(
                f"\nClosing position at {aapl.close[i]} on {str(aapl.index[i])[:10]}",
                attrs=["bold"],
            )
        )
        in_position = False

    earning = round(equity - investment, 2)
    roi = round(earning / investment * 100, 2)
    print(cl(f"EARNING: ${earning} ; ROI: {roi}%", attrs=["bold"]))


implement_strategy(aapl, 100000)


spy = get_historical_data("SPY", "1993-01-01", "1W")
spy_ret = round(((spy.close.iloc[-1] - spy.close.iloc[0]) / spy.close.iloc[0]) * 100)

print(cl("SPY ETF buy/hold return:", attrs=["bold"]), f"{spy_ret}%")
