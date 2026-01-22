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
class TestShareSnapshotAccessAllow(TestShareSnapshot):

    def setUp(self):
        super(TestShareSnapshotAccessAllow, self).setUp()
        self.share_snapshot = manila_fakes.FakeShareSnapshot.create_one_snapshot()
        self.snapshots_mock.get.return_value = self.share_snapshot
        self.access_rule = manila_fakes.FakeSnapshotAccessRule.create_one_access_rule()
        self.snapshots_mock.allow.return_value = self.access_rule._info
        self.cmd = osc_share_snapshots.ShareSnapshotAccessAllow(self.app, None)

    def test_share_snapshot_access_allow(self):
        arglist = [self.share_snapshot.id, 'user', 'demo']
        verifylist = [('snapshot', self.share_snapshot.id), ('access_type', 'user'), ('access_to', 'demo')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.allow.assert_called_with(snapshot=self.share_snapshot, access_type='user', access_to='demo')
        self.assertEqual(tuple(self.access_rule._info.keys()), columns)
        self.assertCountEqual(self.access_rule._info.values(), data)

    def test_share_snapshot_access_allow_exception(self):
        arglist = [self.share_snapshot.id, 'user', 'demo']
        verifylist = [('snapshot', self.share_snapshot.id), ('access_type', 'user'), ('access_to', 'demo')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.snapshots_mock.allow.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)