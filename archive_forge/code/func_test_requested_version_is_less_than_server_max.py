from unittest import mock
import ddt
import manilaclient
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient import exceptions
from manilaclient.tests.unit import utils
def test_requested_version_is_less_than_server_max(self):
    self._mock_returned_server_version('2.17', '2.14')
    max_version = api_versions.APIVersion('2.15')
    manilaclient.API_MAX_VERSION = max_version
    manilaclient.API_MIN_VERSION = api_versions.APIVersion('2.12')
    version = api_versions.discover_version(self.fake_client, max_version)
    self.assertEqual(api_versions.APIVersion('2.15'), version)