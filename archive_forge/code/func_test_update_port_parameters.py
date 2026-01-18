from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests.unit import base
def test_update_port_parameters(self):
    """Test that we detect invalid arguments passed to update_port"""
    self.assertRaises(TypeError, self.cloud.update_port, name_or_id='test-port-id', nome='test-port-name-updated')