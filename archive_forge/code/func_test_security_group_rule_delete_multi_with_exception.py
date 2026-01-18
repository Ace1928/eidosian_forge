from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_security_group_rule_delete_multi_with_exception(self, sgr_mock):
    arglist = [self._security_group_rules[0]['id'], 'unexist_rule']
    verifylist = [('rule', [self._security_group_rules[0]['id'], 'unexist_rule'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    find_mock_result = [None, exceptions.CommandError]
    sgr_mock.side_effect = find_mock_result
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 rules failed to delete.', str(e))
    sgr_mock.assert_any_call(self._security_group_rules[0]['id'])
    sgr_mock.assert_any_call('unexist_rule')