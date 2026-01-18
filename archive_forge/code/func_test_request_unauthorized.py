import abc
from oslo_utils import uuidutils
import osprofiler.profiler
import osprofiler.web
from requests_mock.contrib import fixture as mock_fixture
import testtools
from neutronclient import client
from neutronclient.common import exceptions
def test_request_unauthorized(self):
    text = 'unauthorized message'
    self.requests.register_uri(METHOD, URL, status_code=401, text=text)
    e = self.assertRaises(exceptions.Unauthorized, self.http._cs_request, URL, METHOD)
    self.assertEqual(text, e.message)