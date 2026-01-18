from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests.unit import base
def test_update_port_exception(self):
    port_id = 'd80b1a3b-4fc1-49f3-952e-1e2ab7081d8b'
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports', port_id]), json=self.mock_neutron_port_list_rep), dict(method='PUT', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports', port_id]), status_code=500, validate=dict(json={'port': {'name': 'test-port-name-updated'}}))])
    self.assertRaises(exceptions.SDKException, self.cloud.update_port, name_or_id='d80b1a3b-4fc1-49f3-952e-1e2ab7081d8b', name='test-port-name-updated')
    self.assert_calls()