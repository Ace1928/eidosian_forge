from unittest import mock
import ddt
from oslo_utils import uuidutils
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.v2 import client
@mock.patch.object(client.Client, '_get_keystone_client', mock.Mock())
def test_nonexistent_region_name(self):
    kc = client.Client._get_keystone_client.return_value
    kc.service_catalog = mock.Mock()
    kc.service_catalog.get_endpoints = mock.Mock(return_value=self.catalog)
    self.assertRaises(RuntimeError, client.Client, api_version=manilaclient.API_MAX_VERSION, region_name='FakeRegion')
    self.assertTrue(client.Client._get_keystone_client.called)
    kc.service_catalog.get_endpoints.assert_called_with('sharev2')