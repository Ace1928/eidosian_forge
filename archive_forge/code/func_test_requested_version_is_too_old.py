from unittest import mock
import ddt
import manilaclient
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient import exceptions
from manilaclient.tests.unit import utils
def test_requested_version_is_too_old(self):
    self._mock_returned_server_version('2.5', '2.0')
    manilaclient.API_MAX_VERSION = api_versions.APIVersion('2.5')
    manilaclient.API_MIN_VERSION = api_versions.APIVersion('2.5')
    self.assertRaisesRegex(exceptions.UnsupportedVersion, ".*range is '2.0' to '2.5'.*", api_versions.discover_version, self.fake_client, api_versions.APIVersion('1.0'))