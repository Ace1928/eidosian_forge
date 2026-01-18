import copy
from unittest import mock
from openstackclient.identity.v3 import role_assignment
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_role_assignment_list_inherited(self):
    fake_assignment_a = copy.deepcopy(identity_fakes.ASSIGNMENT_WITH_PROJECT_ID_AND_USER_ID_INHERITED)
    fake_assignment_b = copy.deepcopy(identity_fakes.ASSIGNMENT_WITH_DOMAIN_ID_AND_USER_ID_INHERITED)
    self.role_assignments_mock.list.return_value = [fakes.FakeResource(None, fake_assignment_a, loaded=True), fakes.FakeResource(None, fake_assignment_b, loaded=True)]
    arglist = ['--inherited']
    verifylist = [('user', None), ('group', None), ('system', None), ('domain', None), ('project', None), ('role', None), ('effective', False), ('inherited', True), ('names', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.role_assignments_mock.list.assert_called_with(domain=None, system=None, group=None, effective=False, project=None, role=None, user=None, os_inherit_extension_inherited_to='projects', include_names=False)
    self.assertEqual(self.columns, columns)
    datalist = ((identity_fakes.role_id, identity_fakes.user_id, '', identity_fakes.project_id, '', '', True), (identity_fakes.role_id, identity_fakes.user_id, '', '', identity_fakes.domain_id, '', True))
    self.assertEqual(datalist, tuple(data))