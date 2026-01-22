from __future__ import print_function
from io import StringIO
import json as _json
import warnings
from typing import Optional, Union
from urllib.parse import quote as urlencode
import pandas as pd
import requests
from . import utils, cache
from .data import YfData
from .scrapers.analysis import Analysis
from .scrapers.fundamentals import Fundamentals
from .scrapers.holders import Holders
from .scrapers.quote import Quote, FastInfo
from .scrapers.history import PriceHistory
from .const import _BASE_URL_, _ROOT_URL_
def _get_ticker_tz(self, proxy, timeout):
    proxy = proxy or self.proxy
    if self._tz is not None:
        return self._tz
    c = cache.get_tz_cache()
    tz = c.lookup(self.ticker)
    if tz and (not utils.is_valid_timezone(tz)):
        c.store(self.ticker, None)
        tz = None
    if tz is None:
        tz = self._fetch_ticker_tz(proxy, timeout)
        if utils.is_valid_timezone(tz):
            c.store(self.ticker, tz)
        else:
            tz = None
    self._tz = tz
    return tz