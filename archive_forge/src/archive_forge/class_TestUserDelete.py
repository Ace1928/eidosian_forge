from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestUserDelete(TestUser):

    def setUp(self):
        super(TestUserDelete, self).setUp()
        self.users_mock.get.return_value = self.fake_user
        self.users_mock.delete.return_value = None
        self.cmd = user.DeleteUser(self.app, None)

    def test_user_delete_no_options(self):
        arglist = [self.fake_user.id]
        verifylist = [('users', [self.fake_user.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.users_mock.delete.assert_called_with(self.fake_user.id)
        self.assertIsNone(result)

    @mock.patch.object(utils, 'find_resource')
    def test_delete_multi_users_with_exception(self, find_mock):
        find_mock.side_effect = [self.fake_user, exceptions.CommandError]
        arglist = [self.fake_user.id, 'unexist_user']
        verifylist = [('users', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 users failed to delete.', str(e))
        find_mock.assert_any_call(self.users_mock, self.fake_user.id)
        find_mock.assert_any_call(self.users_mock, 'unexist_user')
        self.assertEqual(2, find_mock.call_count)
        self.users_mock.delete.assert_called_once_with(self.fake_user.id)