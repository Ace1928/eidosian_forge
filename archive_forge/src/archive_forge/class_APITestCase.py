import os
import fixtures
from keystoneauth1 import session as keystone_session
from oslo_serialization import jsonutils
from oslotest import base as test
from requests_mock.contrib import fixture as req_fixture
from urllib import parse as urlparse
from designateclient import client
from designateclient.utils import AdapterWithTimeout
class APITestCase(TestCase):
    """Test case base class for all unit tests."""
    TEST_URL = 'http://127.0.0.1:9001/'
    VERSION = None

    def setUp(self):
        """Run before each test method to initialize test environment."""
        super(TestCase, self).setUp()
        self.log_fixture = self.useFixture(fixtures.FakeLogger())
        self.requests = self.useFixture(req_fixture.Fixture())
        self.client = self.get_client()

    def get_base(self, base_url=None):
        if not base_url:
            base_url = f'{self.TEST_URL}v{self.VERSION}'
        return base_url

    def stub_url(self, method, parts=None, base_url=None, json=None, **kwargs):
        base_url = self.get_base(base_url)
        if json:
            kwargs['text'] = jsonutils.dumps(json)
            headers = kwargs.setdefault('headers', {})
            headers['Content-Type'] = 'application/json'
        if parts:
            url = '/'.join([p.strip('/') for p in [base_url] + parts])
        else:
            url = base_url
        url = url.replace('/?', '?')
        self.requests.register_uri(method, url, **kwargs)

    def get_client(self, version=None, session=None):
        version = version or self.VERSION
        session = session or keystone_session.Session()
        adapted = AdapterWithTimeout(session=session, endpoint_override=self.get_base())
        return client.Client(version, session=adapted)

    def assertRequestBodyIs(self, body=None, json=None):
        last_request_body = self.requests.last_request.body
        if json:
            val = jsonutils.loads(last_request_body)
            self.assertEqual(json, val)
        elif body:
            self.assertEqual(body, last_request_body)

    def assertQueryStringIs(self, qs=''):
        """Verify the QueryString matches what is expected.

        The qs parameter should be of the format 'foo=bar&abc=xyz'
        """
        expected = urlparse.parse_qs(qs, keep_blank_values=True)
        parts = urlparse.urlparse(self.requests.last_request.url)
        querystring = urlparse.parse_qs(parts.query, keep_blank_values=True)
        self.assertEqual(expected, querystring)

    def assertQueryStringContains(self, **kwargs):
        """Verify the query string contains the expected parameters.

        This method is used to verify that the query string for the most recent
        request made contains all the parameters provided as ``kwargs``, and
        that the value of each parameter contains the value for the kwarg. If
        the value for the kwarg is an empty string (''), then all that's
        verified is that the parameter is present.

        """
        parts = urlparse.urlparse(self.requests.last_request.url)
        qs = urlparse.parse_qs(parts.query, keep_blank_values=True)
        for k, v in kwargs.items():
            self.assertIn(k, qs)
            self.assertIn(v, qs[k])

    def assertRequestHeaderEqual(self, name, val):
        """Verify that the last request made contains a header and its value

        The request must have already been made.
        """
        headers = self.requests.last_request.headers
        self.assertEqual(val, headers.get(name))