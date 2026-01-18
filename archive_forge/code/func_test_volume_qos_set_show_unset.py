import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_volume_qos_set_show_unset(self):
    """Tests create volume qos, set, unset, show, delete"""
    name = uuid.uuid4().hex
    cmd_output = self.openstack('volume qos create ' + '--consumer front-end ' + '--property Alpha=a ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'volume qos delete ' + name)
    self.assertEqual(name, cmd_output['name'])
    self.assertEqual('front-end', cmd_output['consumer'])
    self.assertEqual({'Alpha': 'a'}, cmd_output['properties'])
    raw_output = self.openstack('volume qos set ' + '--no-property ' + '--property Beta=b ' + '--property Charlie=c ' + name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('volume qos show ' + name, parse_output=True)
    self.assertEqual(name, cmd_output['name'])
    self.assertEqual({'Beta': 'b', 'Charlie': 'c'}, cmd_output['properties'])
    raw_output = self.openstack('volume qos unset ' + '--property Charlie ' + name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('volume qos show ' + name, parse_output=True)
    self.assertEqual(name, cmd_output['name'])
    self.assertEqual({'Beta': 'b'}, cmd_output['properties'])