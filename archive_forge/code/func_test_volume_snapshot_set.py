import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_volume_snapshot_set(self):
    """Test create, set, unset, show, delete volume snapshot"""
    name = uuid.uuid4().hex
    new_name = name + '_'
    cmd_output = self.openstack('volume snapshot create ' + '--volume ' + self.VOLLY + ' --description aaaa ' + '--property Alpha=a ' + name, parse_output=True)
    self.addCleanup(self.wait_for_delete, 'volume snapshot', new_name)
    self.addCleanup(self.openstack, 'volume snapshot delete ' + new_name)
    self.assertEqual(name, cmd_output['name'])
    self.assertEqual(1, cmd_output['size'])
    self.assertEqual('aaaa', cmd_output['description'])
    self.assertEqual({'Alpha': 'a'}, cmd_output['properties'])
    self.wait_for_status('volume snapshot', name, 'available')
    raw_output = self.openstack('volume snapshot set ' + '--name ' + new_name + ' --description bbbb ' + '--property Alpha=c ' + '--property Beta=b ' + name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('volume snapshot show ' + new_name, parse_output=True)
    self.assertEqual(new_name, cmd_output['name'])
    self.assertEqual(1, cmd_output['size'])
    self.assertEqual('bbbb', cmd_output['description'])
    self.assertEqual({'Alpha': 'c', 'Beta': 'b'}, cmd_output['properties'])
    raw_output = self.openstack('volume snapshot unset ' + '--property Alpha ' + new_name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('volume snapshot show ' + new_name, parse_output=True)
    self.assertEqual({'Beta': 'b'}, cmd_output['properties'])
    raw_output = self.openstack('volume snapshot set ' + '--no-property ' + new_name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('volume snapshot show ' + new_name, parse_output=True)
    self.assertNotIn({'Beta': 'b'}, cmd_output['properties'])