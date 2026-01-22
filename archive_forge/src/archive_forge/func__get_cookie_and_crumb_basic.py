import functools
from functools import lru_cache
import requests as requests
from bs4 import BeautifulSoup
import datetime
from frozendict import frozendict
from . import utils, cache
import threading
@utils.log_indent_decorator
def _get_cookie_and_crumb_basic(self, proxy, timeout):
    cookie = self._get_cookie_basic(proxy, timeout)
    crumb = self._get_crumb_basic(proxy, timeout)
    return (cookie, crumb)