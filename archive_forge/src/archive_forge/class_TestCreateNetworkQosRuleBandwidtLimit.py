from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateNetworkQosRuleBandwidtLimit(TestNetworkQosRule):

    def test_check_type_parameters(self):
        pass

    def setUp(self):
        super(TestCreateNetworkQosRuleBandwidtLimit, self).setUp()
        attrs = {'qos_policy_id': self.qos_policy.id, 'type': RULE_TYPE_BANDWIDTH_LIMIT}
        self.new_rule = network_fakes.FakeNetworkQosRule.create_one_qos_rule(attrs)
        self.columns = ('direction', 'id', 'max_burst_kbits', 'max_kbps', 'project_id', 'qos_policy_id', 'type')
        self.data = (self.new_rule.direction, self.new_rule.id, self.new_rule.max_burst_kbits, self.new_rule.max_kbps, self.new_rule.project_id, self.new_rule.qos_policy_id, self.new_rule.type)
        self.network_client.create_qos_bandwidth_limit_rule = mock.Mock(return_value=self.new_rule)
        self.cmd = network_qos_rule.CreateNetworkQosRule(self.app, self.namespace)

    def test_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_default_options(self):
        arglist = ['--type', RULE_TYPE_BANDWIDTH_LIMIT, '--max-kbps', str(self.new_rule.max_kbps), '--egress', self.new_rule.qos_policy_id]
        verifylist = [('type', RULE_TYPE_BANDWIDTH_LIMIT), ('max_kbps', self.new_rule.max_kbps), ('egress', True), ('qos_policy', self.new_rule.qos_policy_id)]
        rule = network_fakes.FakeNetworkQosRule.create_one_qos_rule({'qos_policy_id': self.qos_policy.id, 'type': RULE_TYPE_BANDWIDTH_LIMIT})
        rule.max_burst_kbits = 0
        expected_data = (rule.direction, rule.id, rule.max_burst_kbits, rule.max_kbps, rule.project_id, rule.qos_policy_id, rule.type)
        with mock.patch.object(self.network_client, 'create_qos_bandwidth_limit_rule', return_value=rule) as create_qos_bandwidth_limit_rule:
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            columns, data = self.cmd.take_action(parsed_args)
        create_qos_bandwidth_limit_rule.assert_called_once_with(self.qos_policy.id, **{'max_kbps': self.new_rule.max_kbps, 'direction': self.new_rule.direction})
        self.assertEqual(self.columns, columns)
        self.assertEqual(expected_data, data)

    def test_create_all_options(self):
        arglist = ['--type', RULE_TYPE_BANDWIDTH_LIMIT, '--max-kbps', str(self.new_rule.max_kbps), '--max-burst-kbits', str(self.new_rule.max_burst_kbits), '--egress', self.new_rule.qos_policy_id]
        verifylist = [('type', RULE_TYPE_BANDWIDTH_LIMIT), ('max_kbps', self.new_rule.max_kbps), ('max_burst_kbits', self.new_rule.max_burst_kbits), ('egress', True), ('qos_policy', self.new_rule.qos_policy_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_qos_bandwidth_limit_rule.assert_called_once_with(self.qos_policy.id, **{'max_kbps': self.new_rule.max_kbps, 'max_burst_kbps': self.new_rule.max_burst_kbits, 'direction': self.new_rule.direction})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_wrong_options(self):
        arglist = ['--type', RULE_TYPE_BANDWIDTH_LIMIT, '--min-kbps', '10000', self.new_rule.qos_policy_id]
        verifylist = [('type', RULE_TYPE_BANDWIDTH_LIMIT), ('min_kbps', 10000), ('qos_policy', self.new_rule.qos_policy_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
        except exceptions.CommandError as e:
            msg = 'Failed to create Network QoS rule: "Create" rule command for type "bandwidth-limit" requires arguments: max_kbps'
            self.assertEqual(msg, str(e))