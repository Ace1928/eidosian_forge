from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_users
from troveclient.tests.osc.v1 import fakes
class TestDatabaseUserShowAccess(TestUsers):
    columns = database_users.ShowDatabaseUserAccess.columns
    values = [('db_1',), ('db_2',)]

    def setUp(self):
        super(TestDatabaseUserShowAccess, self).setUp()
        self.cmd = database_users.ShowDatabaseUserAccess(self.app, None)
        self.data = self.fake_users.get_instances_1234_users_access()
        self.user_client.list_access.return_value = self.data

    @mock.patch.object(utils, 'find_resource')
    def test_user_grant_access(self, mock_find):
        args = ['userinstance', 'user1', '--host', '1.1.1.1']
        verifylist = [('instance', 'userinstance'), ('name', 'user1'), ('host', '1.1.1.1')]
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)