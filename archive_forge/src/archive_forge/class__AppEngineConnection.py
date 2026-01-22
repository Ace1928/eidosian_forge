import requests
import warnings
from requests import adapters
from requests import sessions
from .. import exceptions as exc
from .._compat import gaecontrib
from .._compat import timeout
class _AppEngineConnection(object):
    """Implements urllib3's HTTPConnectionPool API's urlopen().

    This Connection's urlopen() is called with a host-relative path,
    so in order to properly support opening the URL, we need to store
    the full URL when this Connection is constructed from the PoolManager.

    This code wraps AppEngineManager.urlopen(), which exposes a different
    API than in the original urllib3 urlopen(), and thus needs this adapter.
    """

    def __init__(self, appengine_manager, url):
        self.appengine_manager = appengine_manager
        self.url = url

    def urlopen(self, method, url, body=None, headers=None, retries=None, redirect=True, assert_same_host=True, timeout=timeout.Timeout.DEFAULT_TIMEOUT, pool_timeout=None, release_conn=None, **response_kw):
        if not timeout.total:
            timeout.total = timeout._read or timeout._connect
        return self.appengine_manager.urlopen(method, self.url, body=body, headers=headers, retries=retries, redirect=redirect, timeout=timeout, **response_kw)