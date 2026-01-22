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
def _mock_session_and_get_client(self, requested_version, server_max_version, server_min_version, endpoint=None):
    sess = self._create_mock_session(requested_version, server_max_version, server_min_version, endpoint)
    return client.Client(session=sess, microversion=requested_version)