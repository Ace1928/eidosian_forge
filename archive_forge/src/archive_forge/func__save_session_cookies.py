import functools
from functools import lru_cache
import requests as requests
from bs4 import BeautifulSoup
import datetime
from frozendict import frozendict
from . import utils, cache
import threading
def _save_session_cookies(self):
    try:
        cache.get_cookie_cache().store('csrf', self._session.cookies)
    except Exception:
        return False
    return True