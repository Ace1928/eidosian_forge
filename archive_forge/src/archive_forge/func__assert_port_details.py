from openstack.network.v2 import floating_ip
from openstack.network.v2 import network
from openstack.network.v2 import port
from openstack.network.v2 import router
from openstack.network.v2 import subnet
from openstack.tests.functional import base
def _assert_port_details(self, port, port_details):
    self.assertEqual(port.name, port_details['name'])
    self.assertEqual(port.network_id, port_details['network_id'])
    self.assertEqual(port.mac_address, port_details['mac_address'])
    self.assertEqual(port.is_admin_state_up, port_details['admin_state_up'])
    self.assertEqual(port.status, port_details['status'])
    self.assertEqual(port.device_id, port_details['device_id'])
    self.assertEqual(port.device_owner, port_details['device_owner'])