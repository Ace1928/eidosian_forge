from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.osc.v2 import share_backups as osc_share_backups
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_backup_list_for_share(self):
    arglist = ['--share', self.share.id]
    verifylist = [('share', self.share.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.backups_mock.list.assert_called_with(detailed=0, search_opts={'offset': None, 'limit': None, 'name': None, 'description': None, 'name~': None, 'description~': None, 'status': None, 'share_id': self.share.id}, sort_key=None, sort_dir=None)
    self.assertEqual(self.columns, columns)
    self.assertEqual(list(self.values), list(data))