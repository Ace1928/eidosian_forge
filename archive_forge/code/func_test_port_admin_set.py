import uuid
from openstackclient.tests.functional.network.v2 import common
def test_port_admin_set(self):
    """Test create, set (as admin), show, delete"""
    json_output = self.openstack('port create --network %s %s' % (self.NETWORK_NAME, self.NAME), parse_output=True)
    id_ = json_output.get('id')
    self.addCleanup(self.openstack, 'port delete %s' % id_)
    raw_output = self.openstack('--os-username admin port set --mac-address 11:22:33:44:55:66 %s' % self.NAME)
    self.assertOutput('', raw_output)
    json_output = self.openstack('port show %s' % self.NAME, parse_output=True)
    self.assertEqual(json_output.get('mac_address'), '11:22:33:44:55:66')