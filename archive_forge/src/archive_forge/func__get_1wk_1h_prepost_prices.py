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
def _get_1wk_1h_prepost_prices(self):
    if self._prices_1wk_1h_prepost is None:
        self._prices_1wk_1h_prepost = self._tkr.history(period='1wk', interval='1h', auto_adjust=False, prepost=True, proxy=self.proxy)
    return self._prices_1wk_1h_prepost