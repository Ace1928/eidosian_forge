from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.osc.v2 import share_backups as osc_share_backups
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_backup_create(self):
    arglist = [self.share.id]
    verifylist = [('share', self.share.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.backups_mock.create.assert_called_with(self.share)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)