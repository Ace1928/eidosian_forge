from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
def test_server_is_too_old(self):
    fake_client = mock.MagicMock()
    fake_client.versions.get_current.return_value = mock.MagicMock(max_version='1.7', min_version='1.4')
    api_versions.MAX_API_VERSION = '1.10'
    api_versions.MIN_API_VERSION = '1.9'
    self.assertRaises(exceptions.UnsupportedVersion, api_versions.discover_version, fake_client, api_versions.APIVersion('1.latest'))