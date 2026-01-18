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
def year_change(self):
    if self._year_change is not None:
        return self._year_change
    prices = self._get_1y_prices(fullDaysOnly=True)
    if prices.shape[0] >= 2:
        self._year_change = (prices['Close'].iloc[-1] - prices['Close'].iloc[0]) / prices['Close'].iloc[0]
        self._year_change = float(self._year_change)
    return self._year_change