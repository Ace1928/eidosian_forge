from openstackclient.tests.functional.network.v2 import common
class NetworkQosRuleTypeTests(common.NetworkTests):
    """Functional tests for Network QoS rule type."""
    AVAILABLE_RULE_TYPES = ['dscp_marking', 'bandwidth_limit']
    ALL_AVAILABLE_RULE_TYPES = ['dscp_marking', 'bandwidth_limit', 'minimum_bandwidth', 'packet_rate_limit', 'minimum_packet_rate']

    def setUp(self):
        super().setUp()
        if not self.is_extension_enabled('qos'):
            self.skipTest('No qos extension present')

    def test_qos_rule_type_list(self):
        cmd_output = self.openstack('network qos rule type list -f json', parse_output=True)
        for rule_type in self.AVAILABLE_RULE_TYPES:
            self.assertIn(rule_type, [x['Type'] for x in cmd_output])

    def test_qos_rule_type_list_all_supported(self):
        if not self.is_extension_enabled('qos-rule-type-filter'):
            self.skipTest('No "qos-rule-type-filter" extension present')
        cmd_output = self.openstack('network qos rule type list --all-supported -f json', parse_output=True)
        for rule_type in self.AVAILABLE_RULE_TYPES:
            self.assertIn(rule_type, [x['Type'] for x in cmd_output])

    def test_qos_rule_type_list_all_rules(self):
        if not self.is_extension_enabled('qos-rule-type-filter'):
            self.skipTest('No "qos-rule-type-filter" extension present')
        cmd_output = self.openstack('network qos rule type list --all-rules -f json', parse_output=True)
        for rule_type in self.ALL_AVAILABLE_RULE_TYPES:
            self.assertIn(rule_type, [x['Type'] for x in cmd_output])

    def test_qos_rule_type_details(self):
        for rule_type in self.AVAILABLE_RULE_TYPES:
            cmd_output = self.openstack('network qos rule type show %s -f json' % rule_type, parse_output=True)
            self.assertEqual(rule_type, cmd_output['rule_type_name'])
            self.assertIn('drivers', cmd_output.keys())