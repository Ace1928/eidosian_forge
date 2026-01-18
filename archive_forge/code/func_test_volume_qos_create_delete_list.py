import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_volume_qos_create_delete_list(self):
    """Test create, list, delete multiple"""
    name1 = uuid.uuid4().hex
    cmd_output = self.openstack('volume qos create ' + name1, parse_output=True)
    self.assertEqual(name1, cmd_output['name'])
    name2 = uuid.uuid4().hex
    cmd_output = self.openstack('volume qos create ' + name2, parse_output=True)
    self.assertEqual(name2, cmd_output['name'])
    cmd_output = self.openstack('volume qos list', parse_output=True)
    names = [x['Name'] for x in cmd_output]
    self.assertIn(name1, names)
    self.assertIn(name2, names)
    del_output = self.openstack('volume qos delete ' + name1 + ' ' + name2)
    self.assertOutput('', del_output)