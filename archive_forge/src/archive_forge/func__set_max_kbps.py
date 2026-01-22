from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
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