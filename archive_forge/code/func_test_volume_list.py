import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_volume_list(self):
    """Test create, list filter"""
    name1 = uuid.uuid4().hex
    cmd_output = self.openstack('volume create ' + '--size 1 ' + name1, parse_output=True)
    self.addCleanup(self.openstack, 'volume delete ' + name1)
    self.assertEqual(1, cmd_output['size'])
    self.wait_for_status('volume', name1, 'available')
    name2 = uuid.uuid4().hex
    cmd_output = self.openstack('volume create ' + '--size 2 ' + name2, parse_output=True)
    self.addCleanup(self.openstack, 'volume delete ' + name2)
    self.assertEqual(2, cmd_output['size'])
    self.wait_for_status('volume', name2, 'available')
    raw_output = self.openstack('volume set ' + '--state error ' + name2)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('volume list ' + '--long', parse_output=True)
    names = [x['Name'] for x in cmd_output]
    self.assertIn(name1, names)
    self.assertIn(name2, names)
    cmd_output = self.openstack('volume list ' + '--status error', parse_output=True)
    names = [x['Name'] for x in cmd_output]
    self.assertNotIn(name1, names)
    self.assertIn(name2, names)