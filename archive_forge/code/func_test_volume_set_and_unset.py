import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_volume_set_and_unset(self):
    """Tests create volume, set, unset, show, delete"""
    name = uuid.uuid4().hex
    new_name = name + '_'
    cmd_output = self.openstack('volume create ' + '--size 1 ' + '--description aaaa ' + '--property Alpha=a ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'volume delete ' + new_name)
    self.assertEqual(name, cmd_output['name'])
    self.assertEqual(1, cmd_output['size'])
    self.assertEqual('aaaa', cmd_output['description'])
    self.assertEqual({'Alpha': 'a'}, cmd_output['properties'])
    self.assertEqual('false', cmd_output['bootable'])
    self.wait_for_status('volume', name, 'available')
    raw_output = self.openstack('volume set ' + '--name ' + new_name + ' --size 2 ' + '--description bbbb ' + '--no-property ' + '--property Beta=b ' + '--property Gamma=c ' + '--image-property a=b ' + '--image-property c=d ' + '--bootable ' + name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('volume show ' + new_name, parse_output=True)
    self.assertEqual(new_name, cmd_output['name'])
    self.assertEqual(2, cmd_output['size'])
    self.assertEqual('bbbb', cmd_output['description'])
    self.assertEqual({'Beta': 'b', 'Gamma': 'c'}, cmd_output['properties'])
    self.assertEqual({'a': 'b', 'c': 'd'}, cmd_output['volume_image_metadata'])
    self.assertEqual('true', cmd_output['bootable'])
    raw_output = self.openstack('volume unset ' + '--property Beta ' + '--image-property a ' + new_name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('volume show ' + new_name, parse_output=True)
    self.assertEqual({'Gamma': 'c'}, cmd_output['properties'])
    self.assertEqual({'c': 'd'}, cmd_output['volume_image_metadata'])