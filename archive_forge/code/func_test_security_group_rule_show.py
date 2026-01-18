import random
from openstackclient.tests.functional.network.v2 import common
def test_security_group_rule_show(self):
    cmd_output = self.openstack('default security group rule show ' + self.DEFAULT_SG_RULE_ID, parse_output=True)
    self.assertEqual(self.DEFAULT_SG_RULE_ID, cmd_output['id'])
    self.assertEqual(self.protocol, cmd_output['protocol'])
    self.assertEqual(self.port, cmd_output['port_range_min'])
    self.assertEqual(self.port, cmd_output['port_range_max'])
    self.assertEqual(self.direction, cmd_output['direction'])