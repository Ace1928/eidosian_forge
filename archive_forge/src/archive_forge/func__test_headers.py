import abc
from oslo_utils import uuidutils
import osprofiler.profiler
import osprofiler.web
from requests_mock.contrib import fixture as mock_fixture
import testtools
from neutronclient import client
from neutronclient.common import exceptions
def _test_headers(self, expected_headers, **kwargs):
    self.requests.register_uri(METHOD, URL, request_headers=expected_headers)
    self.http.request(URL, METHOD, **kwargs)
    self.assertEqual(kwargs.get('body'), self.requests.last_request.body)