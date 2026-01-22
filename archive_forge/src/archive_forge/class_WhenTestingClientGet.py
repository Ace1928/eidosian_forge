from unittest import mock
from keystoneauth1 import session
from requests_mock.contrib import fixture
import testtools
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.exceptions import UnsupportedVersion
from barbicanclient.tests.utils import get_server_supported_versions
from barbicanclient.tests.utils import get_version_endpoint
from barbicanclient.tests.utils import mock_session
from barbicanclient.tests.utils import mock_session_get
from barbicanclient.tests.utils import mock_session_get_endpoint
class WhenTestingClientGet(TestClient):

    def setUp(self):
        super(WhenTestingClientGet, self).setUp()
        self.httpclient = client._HTTPClient(session=self.session, microversion=_DEFAULT_MICROVERSION, endpoint=self.endpoint)
        self.headers = dict()
        self.href = 'http://test_href/'
        self.get_mock = self.responses.get(self.href, json={})

    def test_get_uses_href_as_is(self):
        self.httpclient.get(self.href)
        self.assertEqual(self.href, self.get_mock.last_request.url)

    def test_get_passes_params(self):
        params = {'test': 'test1'}
        self.httpclient.get(self.href, params=params)
        self.assertEqual(self.href, self.get_mock.last_request.url.split('?')[0])
        self.assertEqual(['test1'], self.get_mock.last_request.qs['test'])

    def test_get_includes_accept_header_of_application_json(self):
        self.httpclient.get(self.href)
        self.assertEqual('application/json', self.get_mock.last_request.headers['Accept'])

    def test_get_includes_default_headers(self):
        self.httpclient._default_headers = {'Test-Default-Header': 'test'}
        self.httpclient.get(self.href)
        self.assertEqual('test', self.get_mock.last_request.headers['Test-Default-Header'])

    def test_get_checks_status_code(self):
        self.httpclient._check_status_code = mock.MagicMock()
        self.httpclient.get(self.href)
        self.httpclient._check_status_code.assert_has_calls([])

    def test_get_raw_uses_href_as_is(self):
        self.httpclient._get_raw(self.href, headers=self.headers)
        self.assertEqual(self.href, self.get_mock.last_request.url)

    def test_get_raw_passes_headers(self):
        self.httpclient._get_raw(self.href, headers={'test': 'test'})
        self.assertEqual('test', self.get_mock.last_request.headers['test'])

    def test_get_raw_includes_default_headers(self):
        self.httpclient._default_headers = {'Test-Default-Header': 'test'}
        self.httpclient._get_raw(self.href, headers=self.headers)
        self.assertIn('Test-Default-Header', self.get_mock.last_request.headers)

    def test_get_raw_checks_status_code(self):
        self.httpclient._check_status_code = mock.MagicMock()
        self.httpclient._get_raw(self.href, headers=self.headers)
        self.httpclient._check_status_code.assert_has_calls([])