import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test_update_network_provider(self):
    network_id = 'test-net-id'
    network_name = 'network'
    network = {'id': network_id, 'name': network_name}
    provider_opts = {'physical_network': 'mynet', 'network_type': 'vlan', 'segmentation_id': 'vlan1', 'should_not_be_passed': 1}
    update_network_provider_opts = {'provider:physical_network': 'mynet', 'provider:network_type': 'vlan', 'provider:segmentation_id': 'vlan1'}
    mock_update_rep = copy.copy(self.mock_new_network_rep)
    mock_update_rep.update(update_network_provider_opts)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks', network_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks'], qs_elements=['name=%s' % network_name]), json={'networks': [network]}), dict(method='PUT', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks', network_id]), json={'network': mock_update_rep}, validate=dict(json={'network': update_network_provider_opts}))])
    network = self.cloud.update_network(network_name, provider=provider_opts)
    self._compare_networks(mock_update_rep, network)
    self.assert_calls()