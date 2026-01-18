import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import role
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_role_remove_user_project(self):
    arglist = ['--user', identity_fakes.user_name, '--project', identity_fakes.project_name, identity_fakes.role_name]
    if self._is_inheritance_testcase():
        arglist.append('--inherited')
    verifylist = [('user', identity_fakes.user_name), ('group', None), ('domain', None), ('project', identity_fakes.project_name), ('role', identity_fakes.role_name), ('inherited', self._is_inheritance_testcase())]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'user': identity_fakes.user_id, 'project': identity_fakes.project_id, 'os_inherit_extension_inherited': self._is_inheritance_testcase()}
    self.roles_mock.revoke.assert_called_with(identity_fakes.role_id, **kwargs)
    self.assertIsNone(result)