import socket
from test import resolvesLocalhostFQDN
import pytest
from urllib3 import connection_from_url
from urllib3.exceptions import ClosedPoolError, LocationValueError
from urllib3.poolmanager import PoolKey, PoolManager, key_fn_by_scheme
from urllib3.util import retry, timeout
def test_pools_keyed_with_from_host(self):
    """Assert pools are still keyed correctly with connection_from_host."""
    ssl_kw = {'key_file': '/root/totally_legit.key', 'cert_file': '/root/totally_legit.crt', 'cert_reqs': 'CERT_REQUIRED', 'ca_certs': '/root/path_to_pem', 'ssl_version': 'SSLv23_METHOD'}
    p = PoolManager(5, **ssl_kw)
    conns = [p.connection_from_host('example.com', 443, scheme='https')]
    for k in ssl_kw:
        p.connection_pool_kw[k] = 'newval'
        conns.append(p.connection_from_host('example.com', 443, scheme='https'))
    assert all((x is not y for i, x in enumerate(conns) for j, y in enumerate(conns) if i != j))