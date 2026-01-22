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
@utils.log_indent_decorator
def _fetch_ticker_tz(self, proxy, timeout):
    proxy = proxy or self.proxy
    logger = utils.get_yf_logger()
    params = {'range': '1d', 'interval': '1d'}
    url = f'{_BASE_URL_}/v8/finance/chart/{self.ticker}'
    try:
        data = self._data.cache_get(url=url, params=params, proxy=proxy, timeout=timeout)
        data = data.json()
    except Exception as e:
        logger.error(f"Failed to get ticker '{self.ticker}' reason: {e}")
        return None
    else:
        error = data.get('chart', {}).get('error', None)
        if error:
            logger.debug(f'Got error from yahoo api for ticker {self.ticker}, Error: {error}')
        else:
            try:
                return data['chart']['result'][0]['meta']['exchangeTimezoneName']
            except Exception as err:
                logger.error(f"Could not get exchangeTimezoneName for ticker '{self.ticker}' reason: {err}")
                logger.debug('Got response: ')
                logger.debug('-------------')
                logger.debug(f' {data}')
                logger.debug('-------------')
    return None