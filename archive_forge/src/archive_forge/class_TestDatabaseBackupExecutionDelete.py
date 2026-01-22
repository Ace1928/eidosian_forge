from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient.osc.v1 import database_backups
from troveclient.tests.osc.v1 import fakes
class TestDatabaseBackupExecutionDelete(TestBackups):

    def setUp(self):
        super(TestDatabaseBackupExecutionDelete, self).setUp()
        self.cmd = database_backups.DeleteDatabaseBackupExecution(self.app, None)

    def test_execution_delete(self):
        args = ['execution']
        parsed_args = self.check_parser(self.cmd, args, [])
        result = self.cmd.take_action(parsed_args)
        self.backup_client.execution_delete.assert_called_with('execution')
        self.assertIsNone(result)