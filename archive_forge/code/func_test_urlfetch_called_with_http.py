import httplib
import pytest
import StringIO
from mock import patch
from ..test_no_ssl import TestWithoutSSL
def test_urlfetch_called_with_http(self):
    """Check that URLFetch is used to fetch non-https resources."""
    resp = MockResponse('OK', 200, False, 'http://www.google.com', {'content-type': 'text/plain'})
    fetch_patch = patch('google.appengine.api.urlfetch.fetch', return_value=resp)
    with fetch_patch as fetch_mock:
        import urllib3
        pool = urllib3.HTTPConnectionPool('www.google.com', '80')
        r = pool.request('GET', '/')
        assert r.status == 200, r.data
        assert fetch_mock.call_count == 1