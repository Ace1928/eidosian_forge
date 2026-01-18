from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_user_set_unexist_user(self):
    arglist = ['unexist-user']
    verifylist = [('name', None), ('password', None), ('email', None), ('project', None), ('enable', False), ('disable', False), ('user', 'unexist-user')]
    self.users_mock.get.side_effect = exceptions.NotFound(None)
    self.users_mock.find.side_effect = exceptions.NotFound(None)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)