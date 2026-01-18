import contextlib
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import user
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_user_create_with_multiple_options(self):
    arglist = ['--ignore-password-expiry', '--disable-multi-factor-auth', '--multi-factor-auth-rule', identity_fakes.mfa_opt1, self.user.name]
    verifylist = [('ignore_password_expiry', True), ('disable_multi_factor_auth', True), ('multi_factor_auth_rule', [identity_fakes.mfa_opt1]), ('enable', False), ('disable', False), ('name', self.user.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'name': self.user.name, 'default_project': None, 'description': None, 'domain': None, 'email': None, 'enabled': True, 'options': {'ignore_password_expiry': True, 'multi_factor_auth_enabled': False, 'multi_factor_auth_rules': [['password', 'totp']]}, 'password': None}
    self.users_mock.create.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)