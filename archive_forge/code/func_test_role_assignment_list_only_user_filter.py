import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v2_0 import role_assignment
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_role_assignment_list_only_user_filter(self):
    arglist = ['--user', identity_fakes.user_name]
    verifylist = [('user', identity_fakes.user_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)