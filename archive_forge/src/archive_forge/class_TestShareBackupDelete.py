from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.osc.v2 import share_backups as osc_share_backups
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareBackupDelete(TestShareBackup):

    def setUp(self):
        super(TestShareBackupDelete, self).setUp()
        self.share_backup = manila_fakes.FakeShareBackup.create_one_backup()
        self.backups_mock.get.return_value = self.share_backup
        self.cmd = osc_share_backups.DeleteShareBackup(self.app, None)

    def test_share_backup_delete_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_backup_delete(self):
        arglist = [self.share_backup.id]
        verifylist = [('backup', [self.share_backup.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.backups_mock.delete.assert_called_with(self.share_backup)
        self.assertIsNone(result)

    def test_share_backup_delete_multiple(self):
        share_backups = manila_fakes.FakeShareBackup.create_share_backups(count=2)
        arglist = [share_backups[0].id, share_backups[1].id]
        verifylist = [('backup', [share_backups[0].id, share_backups[1].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertEqual(self.backups_mock.delete.call_count, len(share_backups))
        self.assertIsNone(result)

    def test_share_backup_delete_exception(self):
        arglist = [self.share_backup.id]
        verifylist = [('backup', [self.share_backup.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.backups_mock.delete.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)