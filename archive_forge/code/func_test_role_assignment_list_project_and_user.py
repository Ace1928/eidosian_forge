import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v2_0 import role_assignment
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_role_assignment_list_project_and_user(self):
    self.roles_mock.roles_for_user.return_value = [fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE), loaded=True), fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE_2), loaded=True)]
    arglist = ['--project', identity_fakes.project_name, '--user', identity_fakes.user_name]
    verifylist = [('user', identity_fakes.user_name), ('project', identity_fakes.project_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.roles_mock.roles_for_user.assert_called_with(identity_fakes.user_id, identity_fakes.project_id)
    self.assertEqual(self.columns, columns)
    datalist = ((identity_fakes.role_id, identity_fakes.user_id, identity_fakes.project_id), (identity_fakes.ROLE_2['id'], identity_fakes.user_id, identity_fakes.project_id))
    self.assertEqual(datalist, tuple(data))