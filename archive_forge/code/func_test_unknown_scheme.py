import json
from test import LONG_TIMEOUT
import pytest
from dummyserver.server import HAS_IPV6
from dummyserver.testcase import HTTPDummyServerTestCase, IPv6HTTPDummyServerTestCase
from urllib3.connectionpool import port_by_scheme
from urllib3.exceptions import MaxRetryError, URLSchemeUnknown
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry
def test_unknown_scheme(self):
    with PoolManager() as http:
        unknown_scheme = 'unknown'
        unknown_scheme_url = '%s://host' % unknown_scheme
        with pytest.raises(URLSchemeUnknown) as e:
            r = http.request('GET', unknown_scheme_url)
        assert e.value.scheme == unknown_scheme
        r = http.request('GET', '%s/redirect' % self.base_url, fields={'target': unknown_scheme_url}, redirect=False)
        assert r.status == 303
        assert r.headers.get('Location') == unknown_scheme_url
        with pytest.raises(URLSchemeUnknown) as e:
            r = http.request('GET', '%s/redirect' % self.base_url, fields={'target': unknown_scheme_url})
        assert e.value.scheme == unknown_scheme