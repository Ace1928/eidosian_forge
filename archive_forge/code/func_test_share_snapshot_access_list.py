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
def test_share_snapshot_access_list(self):
    arglist = [self.share_snapshot.id]
    verifylist = [('snapshot', self.share_snapshot.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.snapshots_mock.access_list.assert_called_with(self.share_snapshot)
    self.assertEqual(self.access_rules_columns, columns)
    self.assertCountEqual(self.values, data)