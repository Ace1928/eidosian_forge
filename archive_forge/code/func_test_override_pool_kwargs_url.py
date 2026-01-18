import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_override_pool_kwargs_url(self):
    """Assert overriding pool kwargs works with connection_from_url."""
    p = PoolManager(strict=True)
    pool_kwargs = {'strict': False, 'retries': 100, 'block': True}
    default_pool = p.connection_from_url('http://example.com/')
    override_pool = p.connection_from_url('http://example.com/', pool_kwargs=pool_kwargs)
    assert default_pool.strict
    assert retry.Retry.DEFAULT == default_pool.retries
    assert not default_pool.block
    assert not override_pool.strict
    assert 100 == override_pool.retries
    assert override_pool.block