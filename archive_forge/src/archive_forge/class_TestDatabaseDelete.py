from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import databases
from troveclient.tests.osc.v1 import fakes
class TestDatabaseDelete(TestDatabases):

    def setUp(self):
        super(TestDatabaseDelete, self).setUp()
        self.cmd = databases.DeleteDatabase(self.app, None)

    @mock.patch.object(utils, 'find_resource')
    def test_database_delete(self, mock_find):
        args = ['instance1', 'db1']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        result = self.cmd.take_action(parsed_args)
        self.database_client.delete.assert_called_with('instance1', 'db1')
        self.assertIsNone(result)

    @mock.patch.object(utils, 'find_resource')
    def test_database_delete_with_exception(self, mock_find):
        args = ['fakeinstance', 'db1']
        parsed_args = self.check_parser(self.cmd, args, [])
        mock_find.side_effect = exceptions.CommandError
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)