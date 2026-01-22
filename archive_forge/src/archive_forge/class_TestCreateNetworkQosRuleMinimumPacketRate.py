from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateNetworkQosRuleMinimumPacketRate(TestNetworkQosRule):

    def test_check_type_parameters(self):
        pass

    def setUp(self):
        super(TestCreateNetworkQosRuleMinimumPacketRate, self).setUp()
        attrs = {'qos_policy_id': self.qos_policy.id, 'type': RULE_TYPE_MINIMUM_PACKET_RATE}
        self.new_rule = network_fakes.FakeNetworkQosRule.create_one_qos_rule(attrs)
        self.columns = ('direction', 'id', 'min_kpps', 'project_id', 'qos_policy_id', 'type')
        self.data = (self.new_rule.direction, self.new_rule.id, self.new_rule.min_kpps, self.new_rule.project_id, self.new_rule.qos_policy_id, self.new_rule.type)
        self.network_client.create_qos_minimum_packet_rate_rule = mock.Mock(return_value=self.new_rule)
        self.cmd = network_qos_rule.CreateNetworkQosRule(self.app, self.namespace)

    def test_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_default_options(self):
        arglist = ['--type', RULE_TYPE_MINIMUM_PACKET_RATE, '--min-kpps', str(self.new_rule.min_kpps), '--egress', self.new_rule.qos_policy_id]
        verifylist = [('type', RULE_TYPE_MINIMUM_PACKET_RATE), ('min_kpps', self.new_rule.min_kpps), ('egress', True), ('qos_policy', self.new_rule.qos_policy_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_qos_minimum_packet_rate_rule.assert_called_once_with(self.qos_policy.id, **{'min_kpps': self.new_rule.min_kpps, 'direction': self.new_rule.direction})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_wrong_options(self):
        arglist = ['--type', RULE_TYPE_MINIMUM_PACKET_RATE, '--min-kbps', '10000', self.new_rule.qos_policy_id]
        verifylist = [('type', RULE_TYPE_MINIMUM_PACKET_RATE), ('min_kbps', 10000), ('qos_policy', self.new_rule.qos_policy_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
        except exceptions.CommandError as e:
            msg = 'Failed to create Network QoS rule: "Create" rule command for type "minimum-packet-rate" requires arguments: direction, min_kpps'
            self.assertEqual(msg, str(e))