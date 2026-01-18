from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share_snapshots as osc_share_snapshots
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_snapshot_show(self):
    arglist = [self.share_snapshot.id]
    verifylist = [('snapshot', self.share_snapshot.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    cliutils.convert_dict_list_to_string = mock.Mock()
    cliutils.convert_dict_list_to_string.return_value = self.export_location
    columns, data = self.cmd.take_action(parsed_args)
    self.snapshots_mock.get.assert_called_with(self.share_snapshot.id)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)