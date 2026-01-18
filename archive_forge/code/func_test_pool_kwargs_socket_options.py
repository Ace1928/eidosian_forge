import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_pool_kwargs_socket_options(self):
    """Assert passing socket options works with connection_from_host"""
    p = PoolManager(socket_options=[])
    override_opts = [(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1), (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
    pool_kwargs = {'socket_options': override_opts}
    default_pool = p.connection_from_host('example.com', scheme='http')
    override_pool = p.connection_from_host('example.com', scheme='http', pool_kwargs=pool_kwargs)
    assert default_pool.conn_kw['socket_options'] == []
    assert override_pool.conn_kw['socket_options'] == override_opts