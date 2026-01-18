import uuid
from openstackclient.tests.functional.network.v2 import common
def test_security_group_show(self):
    cmd_output = self.openstack('security group show ' + self.NAME, parse_output=True)
    self.assertEqual(self.NAME, cmd_output['name'])
    self.assertTrue(cmd_output['stateful'])