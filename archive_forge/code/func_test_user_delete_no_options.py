from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_user_delete_no_options(self):
    arglist = [self.fake_user.id]
    verifylist = [('users', [self.fake_user.id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.users_mock.delete.assert_called_with(self.fake_user.id)
    self.assertIsNone(result)