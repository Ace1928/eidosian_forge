import random
import uuid
from openstackclient.tests.functional.network.v2 import common
def test_subnet_pool_create_delete(self):
    """Test create, delete"""
    name1 = uuid.uuid4().hex
    cmd_output, pool_prefix = self._subnet_pool_create('', name1)
    self.assertEqual(name1, cmd_output['name'])
    self.assertEqual([pool_prefix], cmd_output['prefixes'])
    name2 = uuid.uuid4().hex
    cmd_output, pool_prefix = self._subnet_pool_create('', name2)
    self.assertEqual(name2, cmd_output['name'])
    self.assertEqual([pool_prefix], cmd_output['prefixes'])
    del_output = self.openstack('subnet pool delete ' + name1 + ' ' + name2)
    self.assertOutput('', del_output)