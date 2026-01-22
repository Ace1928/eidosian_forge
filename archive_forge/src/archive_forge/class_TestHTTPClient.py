import abc
from oslo_utils import uuidutils
import osprofiler.profiler
import osprofiler.web
from requests_mock.contrib import fixture as mock_fixture
import testtools
from neutronclient import client
from neutronclient.common import exceptions
class TestHTTPClient(TestHTTPClientMixin, testtools.TestCase):

    def initialize(self):
        return client.HTTPClient(token=AUTH_TOKEN, endpoint_url=END_URL)

    def test_request_error(self):

        def cb(*args, **kwargs):
            raise Exception('error msg')
        self.requests.get(URL, body=cb)
        self.assertRaises(exceptions.ConnectionFailed, self.http._cs_request, URL, METHOD)

    def test_request_success(self):
        text = 'test content'
        self.requests.register_uri(METHOD, URL, text=text)
        resp, resp_text = self.http._cs_request(URL, METHOD)
        self.assertEqual(200, resp.status_code)
        self.assertEqual(text, resp_text)

    def test_request_unauthorized(self):
        text = 'unauthorized message'
        self.requests.register_uri(METHOD, URL, status_code=401, text=text)
        e = self.assertRaises(exceptions.Unauthorized, self.http._cs_request, URL, METHOD)
        self.assertEqual(text, e.message)

    def test_request_forbidden_is_returned_to_caller(self):
        text = 'forbidden message'
        self.requests.register_uri(METHOD, URL, status_code=403, text=text)
        resp, resp_text = self.http._cs_request(URL, METHOD)
        self.assertEqual(403, resp.status_code)
        self.assertEqual(text, resp_text)

    def test_do_request_success(self):
        text = 'test content'
        self.requests.register_uri(METHOD, END_URL + URL, text=text)
        resp, resp_text = self.http.do_request(URL, METHOD)
        self.assertEqual(200, resp.status_code)
        self.assertEqual(text, resp_text)

    def test_do_request_with_headers_success(self):
        text = 'test content'
        self.requests.register_uri(METHOD, END_URL + URL, text=text, request_headers={'key': 'value'})
        resp, resp_text = self.http.do_request(URL, METHOD, headers={'key': 'value'})
        self.assertEqual(200, resp.status_code)
        self.assertEqual(text, resp_text)