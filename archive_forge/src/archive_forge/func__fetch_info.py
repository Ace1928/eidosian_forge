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
def _fetch_info(self, proxy):
    if self._already_fetched:
        return
    self._already_fetched = True
    modules = ['financialData', 'quoteType', 'defaultKeyStatistics', 'assetProfile', 'summaryDetail']
    result = self._fetch(proxy, modules=modules)
    if result is None:
        self._info = {}
        return
    result['quoteSummary']['result'][0]['symbol'] = self._symbol
    query1_info = next((info for info in result.get('quoteSummary', {}).get('result', []) if info['symbol'] == self._symbol), None)
    for k in query1_info:
        if 'maxAge' in query1_info[k] and query1_info[k]['maxAge'] == 1:
            query1_info[k]['maxAge'] = 86400
    query1_info = {k1: v1 for k, v in query1_info.items() if isinstance(v, dict) for k1, v1 in v.items() if v1}

    def _format(k, v):
        if isinstance(v, dict) and 'raw' in v and ('fmt' in v):
            v2 = v['fmt'] if k in {'regularMarketTime', 'postMarketTime'} else v['raw']
        elif isinstance(v, list):
            v2 = [_format(None, x) for x in v]
        elif isinstance(v, dict):
            v2 = {k: _format(k, x) for k, x in v.items()}
        elif isinstance(v, str):
            v2 = v.replace('\xa0', ' ')
        else:
            v2 = v
        return v2
    for k, v in query1_info.items():
        query1_info[k] = _format(k, v)
    self._info = query1_info