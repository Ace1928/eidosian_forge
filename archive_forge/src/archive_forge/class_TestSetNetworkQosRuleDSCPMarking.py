from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetNetworkQosRuleDSCPMarking(TestNetworkQosRule):

    def setUp(self):
        super(TestSetNetworkQosRuleDSCPMarking, self).setUp()
        attrs = {'qos_policy_id': self.qos_policy.id, 'type': RULE_TYPE_DSCP_MARKING}
        self.new_rule = network_fakes.FakeNetworkQosRule.create_one_qos_rule(attrs=attrs)
        self.qos_policy.rules = [self.new_rule]
        self.network_client.update_qos_dscp_marking_rule = mock.Mock(return_value=None)
        self.network_client.find_qos_dscp_marking_rule = mock.Mock(return_value=self.new_rule)
        self.network_client.find_qos_policy = mock.Mock(return_value=self.qos_policy)
        self.cmd = network_qos_rule.SetNetworkQosRule(self.app, self.namespace)

    def test_set_nothing(self):
        arglist = [self.new_rule.qos_policy_id, self.new_rule.id]
        verifylist = [('qos_policy', self.new_rule.qos_policy_id), ('id', self.new_rule.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_qos_dscp_marking_rule.assert_called_with(self.new_rule, self.qos_policy.id)
        self.assertIsNone(result)

    def test_set_dscp_mark(self):
        self._set_dscp_mark()

    def test_set_dscp_mark_to_zero(self):
        self._set_dscp_mark(dscp_mark=0)

    def _set_dscp_mark(self, dscp_mark=None):
        if dscp_mark:
            previous_dscp_mark = self.new_rule.dscp_mark
            self.new_rule.dscp_mark = dscp_mark
        arglist = ['--dscp-mark', str(self.new_rule.dscp_mark), self.new_rule.qos_policy_id, self.new_rule.id]
        verifylist = [('dscp_mark', self.new_rule.dscp_mark), ('qos_policy', self.new_rule.qos_policy_id), ('id', self.new_rule.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'dscp_mark': self.new_rule.dscp_mark}
        self.network_client.update_qos_dscp_marking_rule.assert_called_with(self.new_rule, self.qos_policy.id, **attrs)
        self.assertIsNone(result)
        if dscp_mark:
            self.new_rule.dscp_mark = previous_dscp_mark

    def test_set_wrong_options(self):
        arglist = ['--max-kbps', str(10000), self.new_rule.qos_policy_id, self.new_rule.id]
        verifylist = [('max_kbps', 10000), ('qos_policy', self.new_rule.qos_policy_id), ('id', self.new_rule.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
        except exceptions.CommandError as e:
            msg = 'Failed to set Network QoS rule ID "%(rule)s": Rule type "dscp-marking" only requires arguments: dscp_mark' % {'rule': self.new_rule.id}
            self.assertEqual(msg, str(e))