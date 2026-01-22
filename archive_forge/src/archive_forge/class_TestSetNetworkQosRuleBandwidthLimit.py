from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetNetworkQosRuleBandwidthLimit(TestNetworkQosRule):

    def setUp(self):
        super(TestSetNetworkQosRuleBandwidthLimit, self).setUp()
        attrs = {'qos_policy_id': self.qos_policy.id, 'type': RULE_TYPE_BANDWIDTH_LIMIT}
        self.new_rule = network_fakes.FakeNetworkQosRule.create_one_qos_rule(attrs=attrs)
        self.qos_policy.rules = [self.new_rule]
        self.network_client.update_qos_bandwidth_limit_rule = mock.Mock(return_value=None)
        self.network_client.find_qos_bandwidth_limit_rule = mock.Mock(return_value=self.new_rule)
        self.network_client.find_qos_policy = mock.Mock(return_value=self.qos_policy)
        self.cmd = network_qos_rule.SetNetworkQosRule(self.app, self.namespace)

    def test_set_nothing(self):
        arglist = [self.new_rule.qos_policy_id, self.new_rule.id]
        verifylist = [('qos_policy', self.new_rule.qos_policy_id), ('id', self.new_rule.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_qos_bandwidth_limit_rule.assert_called_with(self.new_rule, self.qos_policy.id)
        self.assertIsNone(result)

    def test_set_max_kbps(self):
        self._set_max_kbps()

    def test_set_max_kbps_to_zero(self):
        self._set_max_kbps(max_kbps=0)

    def _reset_max_kbps(self, max_kbps):
        self.new_rule.max_kbps = max_kbps

    def _set_max_kbps(self, max_kbps=None):
        if max_kbps:
            self.addCleanup(self._reset_max_kbps, self.new_rule.max_kbps)
            self.new_rule.max_kbps = max_kbps
        arglist = ['--max-kbps', str(self.new_rule.max_kbps), self.new_rule.qos_policy_id, self.new_rule.id]
        verifylist = [('max_kbps', self.new_rule.max_kbps), ('qos_policy', self.new_rule.qos_policy_id), ('id', self.new_rule.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'max_kbps': self.new_rule.max_kbps}
        self.network_client.update_qos_bandwidth_limit_rule.assert_called_with(self.new_rule, self.qos_policy.id, **attrs)
        self.assertIsNone(result)

    def test_set_max_burst_kbits(self):
        self._set_max_burst_kbits()

    def test_set_max_burst_kbits_to_zero(self):
        self._set_max_burst_kbits(max_burst_kbits=0)

    def _reset_max_burst_kbits(self, max_burst_kbits):
        self.new_rule.max_burst_kbits = max_burst_kbits

    def _set_max_burst_kbits(self, max_burst_kbits=None):
        if max_burst_kbits:
            self.addCleanup(self._reset_max_burst_kbits, self.new_rule.max_burst_kbits)
            self.new_rule.max_burst_kbits = max_burst_kbits
        arglist = ['--max-burst-kbits', str(self.new_rule.max_burst_kbits), self.new_rule.qos_policy_id, self.new_rule.id]
        verifylist = [('max_burst_kbits', self.new_rule.max_burst_kbits), ('qos_policy', self.new_rule.qos_policy_id), ('id', self.new_rule.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'max_burst_kbps': self.new_rule.max_burst_kbits}
        self.network_client.update_qos_bandwidth_limit_rule.assert_called_with(self.new_rule, self.qos_policy.id, **attrs)
        self.assertIsNone(result)

    def test_set_direction_egress(self):
        self._set_direction('egress')

    def test_set_direction_ingress(self):
        self._set_direction('ingress')

    def _reset_direction(self, direction):
        self.new_rule.direction = direction

    def _set_direction(self, direction):
        self.addCleanup(self._reset_direction, self.new_rule.direction)
        arglist = ['--%s' % direction, self.new_rule.qos_policy_id, self.new_rule.id]
        verifylist = [(direction, True), ('qos_policy', self.new_rule.qos_policy_id), ('id', self.new_rule.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'direction': direction}
        self.network_client.update_qos_bandwidth_limit_rule.assert_called_with(self.new_rule, self.qos_policy.id, **attrs)
        self.assertIsNone(result)

    def test_set_wrong_options(self):
        arglist = ['--min-kbps', str(10000), self.new_rule.qos_policy_id, self.new_rule.id]
        verifylist = [('min_kbps', 10000), ('qos_policy', self.new_rule.qos_policy_id), ('id', self.new_rule.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
        except exceptions.CommandError as e:
            msg = 'Failed to set Network QoS rule ID "%(rule)s": Rule type "bandwidth-limit" only requires arguments: direction, max_burst_kbps, max_kbps' % {'rule': self.new_rule.id}
            self.assertEqual(msg, str(e))