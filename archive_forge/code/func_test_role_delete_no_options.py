from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import role
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_role_delete_no_options(self):
    arglist = [self.fake_role.name]
    verifylist = [('roles', [self.fake_role.name])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.roles_mock.delete.assert_called_with(self.fake_role.id)
    self.assertIsNone(result)