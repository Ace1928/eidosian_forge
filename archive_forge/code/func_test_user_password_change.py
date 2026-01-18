import contextlib
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import user
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_user_password_change(self):
    current_pass = 'old_pass'
    new_pass = 'new_pass'
    arglist = ['--password', new_pass]
    verifylist = [('password', new_pass)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with self._mock_get_password(current_pass):
        result = self.cmd.take_action(parsed_args)
    self.users_mock.update_password.assert_called_with(current_pass, new_pass)
    self.assertIsNone(result)