import random
import uuid
from openstackclient.tests.functional.network.v2 import common
def test_subnet_create_and_delete(self):
    """Test create, delete multiple"""
    name1 = uuid.uuid4().hex
    cmd = 'subnet create --network ' + self.NETWORK_NAME + ' --subnet-range'
    cmd_output = self._subnet_create(cmd, name1)
    self.assertEqual(name1, cmd_output['name'])
    self.assertEqual(self.NETWORK_ID, cmd_output['network_id'])
    name2 = uuid.uuid4().hex
    cmd = 'subnet create --network ' + self.NETWORK_NAME + ' --subnet-range'
    cmd_output = self._subnet_create(cmd, name2)
    self.assertEqual(name2, cmd_output['name'])
    self.assertEqual(self.NETWORK_ID, cmd_output['network_id'])
    del_output = self.openstack('subnet delete ' + name1 + ' ' + name2)
    self.assertOutput('', del_output)