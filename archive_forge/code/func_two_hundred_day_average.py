import datetime
import json
import warnings
from collections.abc import MutableMapping
import numpy as _np
import pandas as pd
import requests
from yfinance import utils
from yfinance.data import YfData
from yfinance.const import quote_summary_valid_modules, _BASE_URL_
from yfinance.exceptions import YFNotImplementedError, YFinanceDataException, YFinanceException
@property
def two_hundred_day_average(self):
    if self._200d_day_average is not None:
        return self._200d_day_average
    prices = self._get_1y_prices(fullDaysOnly=True)
    if prices.empty:
        self._200d_day_average = None
    else:
        n = prices.shape[0]
        a = n - 200
        b = n
        if a < 0:
            a = 0
        self._200d_day_average = float(prices['Close'].iloc[a:b].mean())
    return self._200d_day_average