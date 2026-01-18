import requests
import warnings
from requests import adapters
from requests import sessions
from .. import exceptions as exc
from .._compat import gaecontrib
from .._compat import timeout
def urlopen(self, method, url, body=None, headers=None, retries=None, redirect=True, assert_same_host=True, timeout=timeout.Timeout.DEFAULT_TIMEOUT, pool_timeout=None, release_conn=None, **response_kw):
    if not timeout.total:
        timeout.total = timeout._read or timeout._connect
    return self.appengine_manager.urlopen(method, self.url, body=body, headers=headers, retries=retries, redirect=redirect, timeout=timeout, **response_kw)