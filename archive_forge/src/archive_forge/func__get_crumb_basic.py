import functools
from functools import lru_cache
import requests as requests
from bs4 import BeautifulSoup
import datetime
from frozendict import frozendict
from . import utils, cache
import threading
def _get_crumb_basic(self, proxy=None, timeout=30):
    if self._crumb is not None:
        utils.get_yf_logger().debug('reusing crumb')
        return self._crumb
    cookie = self._get_cookie_basic()
    if cookie is None:
        return None
    get_args = {'url': 'https://query1.finance.yahoo.com/v1/test/getcrumb', 'headers': self.user_agent_headers, 'cookies': {cookie.name: cookie.value}, 'proxies': proxy, 'timeout': timeout, 'allow_redirects': True}
    if self._session_is_caching:
        get_args['expire_after'] = self._expire_after
        crumb_response = self._session.get(**get_args)
    else:
        crumb_response = self._session.get(**get_args)
    self._crumb = crumb_response.text
    if self._crumb is None or '<html>' in self._crumb:
        utils.get_yf_logger().debug("Didn't receive crumb")
        return None
    utils.get_yf_logger().debug(f"crumb = '{self._crumb}'")
    return self._crumb