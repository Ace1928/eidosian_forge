from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_user_set_project(self):
    arglist = ['--project', self.fake_project.id, self.fake_user.name]
    verifylist = [('name', None), ('password', None), ('email', None), ('project', self.fake_project.id), ('enable', False), ('disable', False), ('user', self.fake_user.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.users_mock.update_tenant.assert_called_with(self.fake_user.id, self.fake_project.id)
    self.assertIsNone(result)