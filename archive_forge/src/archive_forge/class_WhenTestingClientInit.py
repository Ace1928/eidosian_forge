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
class WhenTestingClientInit(TestClient):

    def test_api_version_is_appended_to_endpoint(self):
        c = client.Client(session=self.session, endpoint=self.endpoint, project_id=self.project_id)
        self.assertEqual('http://localhost:9311/v1/', c.client.endpoint_override)

    def test_default_headers_are_empty(self):
        c = client._HTTPClient(session=self.session, microversion='1.1', endpoint=self.endpoint)
        self.assertIsInstance(c._default_headers, dict)
        self.assertFalse(bool(c._default_headers))

    def test_project_id_is_added_to_default_headers(self):
        c = client._HTTPClient(session=self.session, microversion=_DEFAULT_MICROVERSION, endpoint=self.endpoint, project_id=self.project_id)
        self.assertIn('X-Project-Id', c._default_headers.keys())
        self.assertEqual(self.project_id, c._default_headers['X-Project-Id'])

    def test_error_thrown_when_no_session_and_no_endpoint(self):
        self.assertRaises(ValueError, client.Client, **{'project_id': self.project_id})

    def test_error_thrown_when_no_session_and_no_project_id(self):
        self.assertRaises(ValueError, client.Client, **{'endpoint': self.endpoint})

    def test_endpoint_override_starts_with_endpoint_url(self):
        c = client.Client(session=self.session, endpoint=self.endpoint, project_id=self.project_id)
        self.assertTrue(c.client.endpoint_override.startswith(self.endpoint))

    def test_endpoint_override_ends_with_default_api_version(self):
        c = client.Client(session=self.session, endpoint=self.endpoint, project_id=self.project_id)
        self.assertTrue(c.client.endpoint_override.rstrip('/').endswith(client._DEFAULT_API_VERSION))