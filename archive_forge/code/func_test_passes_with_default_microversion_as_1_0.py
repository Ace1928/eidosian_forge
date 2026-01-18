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
def test_passes_with_default_microversion_as_1_0(self):
    requested_version = None
    server_max_version = (1, 0)
    server_min_version = (1, 0)
    c = self._mock_session_and_get_client(requested_version, server_max_version, server_min_version)
    self.assertEqual('1.0', c.client.microversion)