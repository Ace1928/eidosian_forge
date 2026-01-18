from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.osc.v2 import share_backups as osc_share_backups
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_backup_delete_multiple(self):
    share_backups = manila_fakes.FakeShareBackup.create_share_backups(count=2)
    arglist = [share_backups[0].id, share_backups[1].id]
    verifylist = [('backup', [share_backups[0].id, share_backups[1].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertEqual(self.backups_mock.delete.call_count, len(share_backups))
    self.assertIsNone(result)