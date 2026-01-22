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
class TestShareSnapshotDelete(TestShareSnapshot):

    def setUp(self):
        super(TestShareSnapshotDelete, self).setUp()
        self.share_snapshot = manila_fakes.FakeShareSnapshot.create_one_snapshot()
        self.snapshots_mock.get.return_value = self.share_snapshot
        self.cmd = osc_share_snapshots.DeleteShareSnapshot(self.app, None)

    def test_share_snapshot_delete_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_snapshot_delete(self):
        arglist = [self.share_snapshot.id]
        verifylist = [('snapshot', [self.share_snapshot.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.snapshots_mock.delete.assert_called_with(self.share_snapshot)
        self.assertIsNone(result)

    def test_share_snapshot_delete_force(self):
        arglist = [self.share_snapshot.id, '--force']
        verifylist = [('snapshot', [self.share_snapshot.id]), ('force', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.snapshots_mock.force_delete.assert_called_with(self.share_snapshot)
        self.assertIsNone(result)

    def test_share_snapshot_delete_multiple(self):
        share_snapshots = manila_fakes.FakeShareSnapshot.create_share_snapshots(count=2)
        arglist = [share_snapshots[0].id, share_snapshots[1].id]
        verifylist = [('snapshot', [share_snapshots[0].id, share_snapshots[1].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertEqual(self.snapshots_mock.delete.call_count, len(share_snapshots))
        self.assertIsNone(result)

    def test_share_snapshot_delete_exception(self):
        arglist = [self.share_snapshot.id]
        verifylist = [('snapshot', [self.share_snapshot.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.snapshots_mock.delete.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_snapshot_delete_wait(self):
        arglist = [self.share_snapshot.id, '--wait']
        verifylist = [('snapshot', [self.share_snapshot.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
            result = self.cmd.take_action(parsed_args)
            self.snapshots_mock.delete.assert_called_with(self.share_snapshot)
            self.assertIsNone(result)

    def test_share_snapshot_delete_wait_error(self):
        arglist = [self.share_snapshot.id, '--wait']
        verifylist = [('snapshot', [self.share_snapshot.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=False):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)