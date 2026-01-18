import uuid
from openstackclient.tests.functional.network.v2 import common
def test_router_create_and_delete(self):
    """Test create options, delete multiple"""
    name1 = uuid.uuid4().hex
    name2 = uuid.uuid4().hex
    cmd_output = self.openstack('router create ' + name1, parse_output=True)
    self.assertEqual(name1, cmd_output['name'])
    cmd_output = self.openstack('router create ' + name2, parse_output=True)
    self.assertEqual(name2, cmd_output['name'])
    del_output = self.openstack('router delete ' + name1 + ' ' + name2)
    self.assertOutput('', del_output)