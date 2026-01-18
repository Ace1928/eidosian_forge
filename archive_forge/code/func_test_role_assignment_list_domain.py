import copy
from unittest import mock
from openstackclient.identity.v3 import role_assignment
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_role_assignment_list_domain(self):
    self.role_assignments_mock.list.return_value = [fakes.FakeResource(None, copy.deepcopy(identity_fakes.ASSIGNMENT_WITH_DOMAIN_ID_AND_USER_ID), loaded=True), fakes.FakeResource(None, copy.deepcopy(identity_fakes.ASSIGNMENT_WITH_DOMAIN_ID_AND_GROUP_ID), loaded=True)]
    arglist = ['--domain', identity_fakes.domain_name]
    verifylist = [('user', None), ('group', None), ('system', None), ('domain', identity_fakes.domain_name), ('project', None), ('role', None), ('effective', False), ('inherited', False), ('names', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.role_assignments_mock.list.assert_called_with(domain=self.domains_mock.get(), system=None, group=None, effective=False, project=None, role=None, user=None, os_inherit_extension_inherited_to=None, include_names=False)
    self.assertEqual(self.columns, columns)
    datalist = ((identity_fakes.role_id, identity_fakes.user_id, '', '', identity_fakes.domain_id, '', False), (identity_fakes.role_id, '', identity_fakes.group_id, '', identity_fakes.domain_id, '', False))
    self.assertEqual(datalist, tuple(data))