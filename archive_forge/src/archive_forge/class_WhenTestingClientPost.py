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
class WhenTestingClientPost(TestClient):

    def setUp(self):
        super(WhenTestingClientPost, self).setUp()
        self.httpclient = client._HTTPClient(session=self.session, microversion=_DEFAULT_MICROVERSION, endpoint=self.endpoint, version='v1')
        self.href = self.endpoint + '/v1/secrets/'
        self.post_mock = self.responses.post(self.href, json={})

    def test_post_normalizes_url_with_traling_slash(self):
        self.httpclient.post(path='secrets', json={'test_data': 'test'})
        self.assertTrue(self.post_mock.last_request.url.endswith('/'))

    def test_post_includes_content_type_header_of_application_json(self):
        self.httpclient.post(path='secrets', json={'test_data': 'test'})
        self.assertEqual('application/json', self.post_mock.last_request.headers['Content-Type'])

    def test_post_includes_default_headers(self):
        self.httpclient._default_headers = {'Test-Default-Header': 'test'}
        self.httpclient.post(path='secrets', json={'test_data': 'test'})
        self.assertEqual('test', self.post_mock.last_request.headers['Test-Default-Header'])

    def test_post_checks_status_code(self):
        self.httpclient._check_status_code = mock.MagicMock()
        self.httpclient.post(path='secrets', json={'test_data': 'test'})
        self.httpclient._check_status_code.assert_has_calls([])