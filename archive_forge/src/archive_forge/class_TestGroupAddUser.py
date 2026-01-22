from unittest import mock
from unittest.mock import call
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestGroupAddUser(TestGroup):
    _group = identity_fakes.FakeGroup.create_one_group()
    users = identity_fakes.FakeUser.create_users(count=2)

    def setUp(self):
        super(TestGroupAddUser, self).setUp()
        self.groups_mock.get.return_value = self._group
        self.users_mock.get = identity_fakes.FakeUser.get_users(self.users)
        self.users_mock.add_to_group.return_value = None
        self.cmd = group.AddUserToGroup(self.app, None)

    def test_group_add_user(self):
        arglist = [self._group.name, self.users[0].name]
        verifylist = [('group', self._group.name), ('user', [self.users[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.users_mock.add_to_group.assert_called_once_with(self.users[0].id, self._group.id)
        self.assertIsNone(result)

    def test_group_add_multi_users(self):
        arglist = [self._group.name, self.users[0].name, self.users[1].name]
        verifylist = [('group', self._group.name), ('user', [self.users[0].name, self.users[1].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = [call(self.users[0].id, self._group.id), call(self.users[1].id, self._group.id)]
        self.users_mock.add_to_group.assert_has_calls(calls)
        self.assertIsNone(result)

    @mock.patch.object(group.LOG, 'error')
    def test_group_add_user_with_error(self, mock_error):
        self.users_mock.add_to_group.side_effect = [exceptions.CommandError(), None]
        arglist = [self._group.name, self.users[0].name, self.users[1].name]
        verifylist = [('group', self._group.name), ('user', [self.users[0].name, self.users[1].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            msg = '1 of 2 users not added to group %s.' % self._group.name
            self.assertEqual(msg, str(e))
        msg = '%(user)s not added to group %(group)s: ' % {'user': self.users[0].name, 'group': self._group.name}
        mock_error.assert_called_once_with(msg)