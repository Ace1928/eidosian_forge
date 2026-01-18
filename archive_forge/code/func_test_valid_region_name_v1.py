from unittest import mock
import ddt
from oslo_utils import uuidutils
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.v2 import client
@mock.patch.object(client.Client, '_get_keystone_client', mock.Mock())
def test_valid_region_name_v1(self):
    self.mock_object(client.httpclient, 'HTTPClient')
    kc = client.Client._get_keystone_client.return_value
    kc.service_catalog = mock.Mock()
    kc.service_catalog.get_endpoints = mock.Mock(return_value=self.catalog)
    c = client.Client(api_version=manilaclient.API_DEPRECATED_VERSION, service_type='share', region_name='TestRegion')
    self.assertTrue(client.Client._get_keystone_client.called)
    kc.service_catalog.get_endpoints.assert_called_with('share')
    client.httpclient.HTTPClient.assert_called_with('http://1.2.3.4', mock.ANY, 'python-manilaclient', insecure=False, cacert=None, cert=None, timeout=None, retries=None, http_log_debug=False, api_version=manilaclient.API_DEPRECATED_VERSION)
    self.assertIsNotNone(c.client)