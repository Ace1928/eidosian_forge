import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_pool_manager_no_url_absolute_form(self):
    """Valides we won't send a request with absolute form without a proxy"""
    p = PoolManager(strict=True)
    assert p._proxy_requires_url_absolute_form('http://example.com') is False
    assert p._proxy_requires_url_absolute_form('https://example.com') is False