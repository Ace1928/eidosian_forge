import random
import uuid
from openstackclient.tests.functional.network.v2 import common
def test_subnet_list(self):
    """Test create, list filter"""
    name1 = uuid.uuid4().hex
    name2 = uuid.uuid4().hex
    cmd = 'subnet create ' + '--network ' + self.NETWORK_NAME + ' --dhcp --subnet-range'
    cmd_output = self._subnet_create(cmd, name1)
    self.addCleanup(self.openstack, 'subnet delete ' + name1)
    self.assertEqual(name1, cmd_output['name'])
    self.assertEqual(True, cmd_output['enable_dhcp'])
    self.assertEqual(self.NETWORK_ID, cmd_output['network_id'])
    self.assertEqual(4, cmd_output['ip_version'])
    cmd = 'subnet create ' + '--network ' + self.NETWORK_NAME + ' --ip-version 6 --no-dhcp ' + '--subnet-range'
    cmd_output = self._subnet_create(cmd, name2, is_type_ipv4=False)
    self.addCleanup(self.openstack, 'subnet delete ' + name2)
    self.assertEqual(name2, cmd_output['name'])
    self.assertEqual(False, cmd_output['enable_dhcp'])
    self.assertEqual(self.NETWORK_ID, cmd_output['network_id'])
    self.assertEqual(6, cmd_output['ip_version'])
    cmd_output = self.openstack('subnet list ' + '--long ', parse_output=True)
    names = [x['Name'] for x in cmd_output]
    self.assertIn(name1, names)
    self.assertIn(name2, names)
    cmd_output = self.openstack('subnet list ' + '--name ' + name1, parse_output=True)
    names = [x['Name'] for x in cmd_output]
    self.assertIn(name1, names)
    self.assertNotIn(name2, names)
    cmd_output = self.openstack('subnet list ' + '--ip-version 6', parse_output=True)
    names = [x['Name'] for x in cmd_output]
    self.assertNotIn(name1, names)
    self.assertIn(name2, names)
    cmd_output = self.openstack('subnet list ' + '--network ' + self.NETWORK_ID, parse_output=True)
    names = [x['Name'] for x in cmd_output]
    self.assertIn(name1, names)
    self.assertIn(name2, names)
    cmd_output = self.openstack('subnet list ' + '--no-dhcp ', parse_output=True)
    names = [x['Name'] for x in cmd_output]
    self.assertNotIn(name1, names)
    self.assertIn(name2, names)