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
class TestShareSnapshotUnset(TestShareSnapshot):

    def setUp(self):
        super(TestShareSnapshotUnset, self).setUp()
        self.share_snapshot = manila_fakes.FakeShareSnapshot.create_one_snapshot(methods={'delete_metadata': None})
        self.snapshots_mock.get.return_value = self.share_snapshot
        self.cmd = osc_share_snapshots.UnsetShareSnapshot(self.app, None)

    def test_unset_snapshot_name(self):
        arglist = [self.share_snapshot.id, '--name']
        verifylist = [('snapshot', self.share_snapshot.id), ('name', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.snapshots_mock.update.assert_called_with(self.share_snapshot, display_name=None)
        self.assertIsNone(result)

    def test_unset_snapshot_description(self):
        arglist = [self.share_snapshot.id, '--description']
        verifylist = [('snapshot', self.share_snapshot.id), ('description', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.snapshots_mock.update.assert_called_with(self.share_snapshot, display_description=None)
        self.assertIsNone(result)

    def test_unset_snapshot_property(self):
        arglist = ['--property', 'Manila', self.share_snapshot.id]
        verifylist = [('property', ['Manila']), ('snapshot', self.share_snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.share_snapshot.delete_metadata.assert_called_with(parsed_args.property)

    def test_unset_snapshot_name_exception(self):
        arglist = [self.share_snapshot.id, '--name']
        verifylist = [('snapshot', self.share_snapshot.id), ('name', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.snapshots_mock.update.side_effect = Exception()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_unset_snapshot_property_exception(self):
        arglist = ['--property', 'Manila', self.share_snapshot.id]
        verifylist = [('property', ['Manila']), ('snapshot', self.share_snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.share_snapshot.delete_metadata.assert_called_with(parsed_args.property)
        self.share_snapshot.delete_metadata.side_effect = exceptions.NotFound
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)