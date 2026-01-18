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
def test_share_snapshot_delete_multiple(self):
    share_snapshots = manila_fakes.FakeShareSnapshot.create_share_snapshots(count=2)
    arglist = [share_snapshots[0].id, share_snapshots[1].id]
    verifylist = [('snapshot', [share_snapshots[0].id, share_snapshots[1].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertEqual(self.snapshots_mock.delete.call_count, len(share_snapshots))
    self.assertIsNone(result)