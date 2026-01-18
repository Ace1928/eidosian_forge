from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_wrong_options(self):
    arglist = ['--min-kbps', str(10000), self.new_rule.qos_policy_id, self.new_rule.id]
    verifylist = [('min_kbps', 10000), ('qos_policy', self.new_rule.qos_policy_id), ('id', self.new_rule.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
    except exceptions.CommandError as e:
        msg = 'Failed to set Network QoS rule ID "%(rule)s": Rule type "bandwidth-limit" only requires arguments: direction, max_burst_kbps, max_kbps' % {'rule': self.new_rule.id}
        self.assertEqual(msg, str(e))