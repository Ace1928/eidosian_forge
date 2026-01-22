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
def _get_exchange_metadata(self):
    if self._md is not None:
        return self._md
    self._get_1y_prices()
    self._md = self._tkr.get_history_metadata(proxy=self.proxy)
    return self._md