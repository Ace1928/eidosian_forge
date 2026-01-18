import json
from test import LONG_TIMEOUT
import pytest
from dummyserver.server import HAS_IPV6
from dummyserver.testcase import HTTPDummyServerTestCase, IPv6HTTPDummyServerTestCase
from urllib3.connectionpool import port_by_scheme
from urllib3.exceptions import MaxRetryError, URLSchemeUnknown
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry
def test_too_many_redirects(self):
    with PoolManager() as http:
        with pytest.raises(MaxRetryError):
            http.request('GET', '%s/redirect' % self.base_url, fields={'target': '%s/redirect?target=%s/' % (self.base_url, self.base_url)}, retries=1, preload_content=False)
        with pytest.raises(MaxRetryError):
            http.request('GET', '%s/redirect' % self.base_url, fields={'target': '%s/redirect?target=%s/' % (self.base_url, self.base_url)}, retries=Retry(total=None, redirect=1), preload_content=False)
        assert len(http.pools) == 1
        pool = http.connection_from_host(self.host, self.port)
        assert pool.num_connections == 1