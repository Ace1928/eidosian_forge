import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import role
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
@mock.patch.object(common, 'find_user')
def test_role_remove_non_existent_user_domain(self, find_mock):
    find_mock.side_effect = exceptions.CommandError
    arglist = ['--user', identity_fakes.user_id, '--domain', identity_fakes.domain_name, identity_fakes.role_name]
    if self._is_inheritance_testcase():
        arglist.append('--inherited')
    verifylist = [('user', identity_fakes.user_id), ('group', None), ('system', None), ('domain', identity_fakes.domain_name), ('project', None), ('role', identity_fakes.role_name), ('inherited', self._is_inheritance_testcase())]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'user': identity_fakes.user_id, 'domain': identity_fakes.domain_id, 'os_inherit_extension_inherited': self._is_inheritance_testcase()}
    self.roles_mock.revoke.assert_called_with(identity_fakes.role_id, **kwargs)
    self.assertIsNone(result)