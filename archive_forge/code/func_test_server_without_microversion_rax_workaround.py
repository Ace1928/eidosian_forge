from unittest import mock
import novaclient
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient import utils as nutils
from novaclient.v2 import versions
def test_server_without_microversion_rax_workaround(self):
    fake_client = mock.MagicMock()
    fake_client.versions.get_current.return_value = None
    novaclient.API_MAX_VERSION = api_versions.APIVersion('2.11')
    novaclient.API_MIN_VERSION = api_versions.APIVersion('2.1')
    self.assertEqual('2.0', api_versions.discover_version(fake_client, api_versions.APIVersion('2.latest')).get_string())