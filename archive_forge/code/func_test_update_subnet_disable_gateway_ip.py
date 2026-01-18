import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import subnet as _subnet
from openstack.tests.unit import base
def test_update_subnet_disable_gateway_ip(self):
    expected_subnet = copy.copy(self.mock_subnet_rep)
    expected_subnet['gateway_ip'] = None
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets', self.subnet_id]), json=self.mock_subnet_rep), dict(method='PUT', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets', self.subnet_id]), json={'subnet': expected_subnet}, validate=dict(json={'subnet': {'gateway_ip': None}}))])
    subnet = self.cloud.update_subnet(self.subnet_id, disable_gateway_ip=True)
    self._compare_subnets(expected_subnet, subnet)
    self.assert_calls()