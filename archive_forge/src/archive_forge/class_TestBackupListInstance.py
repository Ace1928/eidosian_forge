from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient.osc.v1 import database_backups
from troveclient.tests.osc.v1 import fakes
class TestBackupListInstance(TestBackups):
    defaults = {'limit': None, 'marker': None}
    columns = database_backups.ListDatabaseInstanceBackups.columns
    values = ('bk-1234', '1234', 'bkp_1', 'COMPLETED', None, '2015-05-16T14:23:08')

    def setUp(self):
        super(TestBackupListInstance, self).setUp()
        self.cmd = database_backups.ListDatabaseInstanceBackups(self.app, None)
        data = [self.fake_backups.get_backup_bk_1234()]
        self.instance_client.backups.return_value = common.Paginated(data)

    @mock.patch.object(utils, 'find_resource')
    def test_backup_list_defaults(self, mock_find):
        args = ['1234']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.instance_client.backups.assert_called_once_with('1234', **self.defaults)
        self.assertEqual(self.columns, columns)
        self.assertEqual([self.values], data)