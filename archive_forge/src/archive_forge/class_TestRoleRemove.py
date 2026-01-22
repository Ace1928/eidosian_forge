from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import role
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestRoleRemove(TestRole):

    def setUp(self):
        super(TestRoleRemove, self).setUp()
        self.projects_mock.get.return_value = self.fake_project
        self.users_mock.get.return_value = self.fake_user
        self.roles_mock.get.return_value = self.fake_role
        self.roles_mock.remove_user_role.return_value = None
        self.cmd = role.RemoveRole(self.app, None)

    def test_role_remove(self):
        arglist = ['--project', self.fake_project.name, '--user', self.fake_user.name, self.fake_role.name]
        verifylist = [('role', self.fake_role.name), ('project', self.fake_project.name), ('user', self.fake_user.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.roles_mock.remove_user_role.assert_called_with(self.fake_user.id, self.fake_role.id, self.fake_project.id)
        self.assertIsNone(result)