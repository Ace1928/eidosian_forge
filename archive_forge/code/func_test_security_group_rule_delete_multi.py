from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_security_group_rule_delete_multi(self, sgr_mock):
    arglist = []
    verifylist = []
    for s in self._security_group_rules:
        arglist.append(s['id'])
    verifylist = [('rule', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for s in self._security_group_rules:
        calls.append(call(s['id']))
    sgr_mock.assert_has_calls(calls)
    self.assertIsNone(result)