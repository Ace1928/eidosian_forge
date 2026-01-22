import functools
from functools import lru_cache
import requests as requests
from bs4 import BeautifulSoup
import datetime
from frozendict import frozendict
from . import utils, cache
import threading
def _load_session_cookies(self):
    cookie_dict = cache.get_cookie_cache().lookup('csrf')
    if cookie_dict is None:
        return False
    if cookie_dict['age'] > datetime.timedelta(days=1):
        return False
    self._session.cookies.update(cookie_dict['cookie'])
    utils.get_yf_logger().debug('loaded persistent cookie')