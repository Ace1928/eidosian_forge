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
def year_high(self):
    if self._year_high is not None:
        return self._year_high
    prices = self._get_1y_prices(fullDaysOnly=True)
    if prices.empty:
        prices = self._get_1y_prices(fullDaysOnly=False)
    self._year_high = float(prices['High'].max())
    return self._year_high