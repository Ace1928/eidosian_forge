import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import role
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_role_add_domain_role_on_user_project(self):
    self.roles_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE_2), loaded=True)
    arglist = ['--user', identity_fakes.user_name, '--project', identity_fakes.project_name, '--role-domain', identity_fakes.domain_name, identity_fakes.ROLE_2['name']]
    if self._is_inheritance_testcase():
        arglist.append('--inherited')
    verifylist = [('user', identity_fakes.user_name), ('group', None), ('domain', None), ('project', identity_fakes.project_name), ('role', identity_fakes.ROLE_2['name']), ('inherited', self._is_inheritance_testcase())]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'user': identity_fakes.user_id, 'project': identity_fakes.project_id, 'os_inherit_extension_inherited': self._is_inheritance_testcase()}
    self.roles_mock.grant.assert_called_with(identity_fakes.ROLE_2['id'], **kwargs)
    self.assertIsNone(result)