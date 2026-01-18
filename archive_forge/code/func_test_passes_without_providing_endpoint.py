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
def test_passes_without_providing_endpoint(self):
    requested_version = None
    server_max_version = (1, 1)
    server_min_version = (1, 0)
    endpoint = None
    self._test_client_creation_with_endpoint(requested_version, server_max_version, server_min_version, endpoint)