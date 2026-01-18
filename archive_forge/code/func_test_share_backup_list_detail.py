from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.osc.v2 import share_backups as osc_share_backups
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_backup_list_detail(self):
    arglist = ['--detail']
    verifylist = [('detail', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.backups_mock.list.assert_called_with(detailed=1, search_opts={'offset': None, 'limit': None, 'name': None, 'description': None, 'name~': None, 'description~': None, 'status': None, 'share_id': None}, sort_key=None, sort_dir=None)
    self.assertEqual(self.detailed_columns, columns)
    self.assertEqual(list(self.detailed_values), list(data))