from openstackclient.tests.functional.network.v2 import common
def test_qos_rule_type_details(self):
    for rule_type in self.AVAILABLE_RULE_TYPES:
        cmd_output = self.openstack('network qos rule type show %s -f json' % rule_type, parse_output=True)
        self.assertEqual(rule_type, cmd_output['rule_type_name'])
        self.assertIn('drivers', cmd_output.keys())