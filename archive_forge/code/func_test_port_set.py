import uuid
from openstackclient.tests.functional.network.v2 import common
def test_port_set(self):
    """Test create, set, show, delete"""
    name = uuid.uuid4().hex
    json_output = self.openstack('port create --network %s --description xyzpdq --disable %s' % (self.NETWORK_NAME, name), parse_output=True)
    id1 = json_output.get('id')
    self.addCleanup(self.openstack, 'port delete %s' % id1)
    self.assertEqual(name, json_output.get('name'))
    self.assertEqual('xyzpdq', json_output.get('description'))
    self.assertEqual(False, json_output.get('admin_state_up'))
    raw_output = self.openstack('port set --enable %s' % name)
    self.assertOutput('', raw_output)
    json_output = self.openstack('port show %s' % name, parse_output=True)
    sg_id = json_output.get('security_group_ids')[0]
    self.assertEqual(name, json_output.get('name'))
    self.assertEqual('xyzpdq', json_output.get('description'))
    self.assertEqual(True, json_output.get('admin_state_up'))
    self.assertIsNotNone(json_output.get('mac_address'))
    raw_output = self.openstack('port unset --security-group %s %s' % (sg_id, id1))
    self.assertOutput('', raw_output)
    json_output = self.openstack('port show %s' % name, parse_output=True)
    self.assertEqual([], json_output.get('security_group_ids'))