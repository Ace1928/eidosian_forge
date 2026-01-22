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
class WhenTestingClientPut(TestClient):

    def setUp(self):
        super(WhenTestingClientPut, self).setUp()
        self.httpclient = client._HTTPClient(session=self.session, microversion=_DEFAULT_MICROVERSION, endpoint=self.endpoint)
        self.href = 'http://test_href/'
        self.put_mock = self.responses.put(self.href, status_code=204)

    def test_put_uses_href_as_is(self):
        self.httpclient.put(self.href)
        self.assertTrue(self.put_mock.called)

    def test_put_passes_data(self):
        data = 'test'
        self.httpclient.put(self.href, data=data)
        self.assertEqual('test', self.put_mock.last_request.text)

    def test_put_includes_default_headers(self):
        self.httpclient._default_headers = {'Test-Default-Header': 'test'}
        self.httpclient.put(self.href)
        self.assertEqual('test', self.put_mock.last_request.headers['Test-Default-Header'])

    def test_put_checks_status_code(self):
        self.httpclient._check_status_code = mock.MagicMock()
        self.httpclient.put(self.href, data='test')
        self.httpclient._check_status_code.assert_has_calls([])