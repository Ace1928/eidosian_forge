from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_qos_policy_delete_error(self):
    arglist = [self.new_rule.qos_policy_id, self.new_rule.id]
    verifylist = [('qos_policy', self.new_rule.qos_policy_id), ('id', self.new_rule.id)]
    self.network_client.delete_qos_bandwidth_limit_rule.side_effect = Exception('Error message')
    try:
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
    except exceptions.CommandError as e:
        msg = 'Failed to delete Network QoS rule ID "%(rule)s": %(e)s' % {'rule': self.new_rule.id, 'e': 'Error message'}
        self.assertEqual(msg, str(e))