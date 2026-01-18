import uuid
from openstackclient.tests.functional.volume.v2 import common
def test_volume_snapshot_delete(self):
    """Test create, delete multiple"""
    name1 = uuid.uuid4().hex
    cmd_output = self.openstack('volume snapshot create ' + name1 + ' --volume ' + self.VOLLY, parse_output=True)
    self.assertEqual(name1, cmd_output['name'])
    name2 = uuid.uuid4().hex
    cmd_output = self.openstack('volume snapshot create ' + name2 + ' --volume ' + self.VOLLY, parse_output=True)
    self.assertEqual(name2, cmd_output['name'])
    self.wait_for_status('volume snapshot', name1, 'available')
    self.wait_for_status('volume snapshot', name2, 'available')
    del_output = self.openstack('volume snapshot delete ' + name1 + ' ' + name2)
    self.assertOutput('', del_output)
    self.wait_for_delete('volume snapshot', name1)
    self.wait_for_delete('volume snapshot', name2)