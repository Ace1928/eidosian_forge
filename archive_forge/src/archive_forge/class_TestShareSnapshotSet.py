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
class TestShareSnapshotSet(TestShareSnapshot):

    def setUp(self):
        super(TestShareSnapshotSet, self).setUp()
        self.share_snapshot = manila_fakes.FakeShareSnapshot.create_one_snapshot(methods={'set_metadata': None})
        self.snapshots_mock.get.return_value = self.share_snapshot
        self.cmd = osc_share_snapshots.SetShareSnapshot(self.app, None)

    def test_set_snapshot_name(self):
        snapshot_name = 'snapshot-name-' + uuid.uuid4().hex
        arglist = [self.share_snapshot.id, '--name', snapshot_name]
        verifylist = [('snapshot', self.share_snapshot.id), ('name', snapshot_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.snapshots_mock.update.assert_called_with(self.share_snapshot, display_name=parsed_args.name)
        self.assertIsNone(result)

    def test_set_snapshot_description(self):
        description = 'snapshot-description-' + uuid.uuid4().hex
        arglist = [self.share_snapshot.id, '--description', description]
        verifylist = [('snapshot', self.share_snapshot.id), ('description', description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.snapshots_mock.update.assert_called_with(self.share_snapshot, display_description=parsed_args.description)
        self.assertIsNone(result)

    def test_set_snapshot_status(self):
        arglist = [self.share_snapshot.id, '--status', 'available']
        verifylist = [('snapshot', self.share_snapshot.id), ('status', 'available')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.snapshots_mock.reset_state.assert_called_with(self.share_snapshot, parsed_args.status)
        self.assertIsNone(result)

    def test_set_snapshot_property(self):
        arglist = [self.share_snapshot.id, '--property', 'Zorilla=manila']
        verifylist = [('snapshot', self.share_snapshot.id), ('property', {'Zorilla': 'manila'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.share_snapshot.set_metadata.assert_called_with({'Zorilla': 'manila'})

    def test_set_snapshot_update_exception(self):
        snapshot_name = 'snapshot-name-' + uuid.uuid4().hex
        arglist = [self.share_snapshot.id, '--name', snapshot_name]
        verifylist = [('snapshot', self.share_snapshot.id), ('name', snapshot_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.snapshots_mock.update.side_effect = Exception()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_set_snapshot_status_exception(self):
        arglist = [self.share_snapshot.id, '--status', 'available']
        verifylist = [('snapshot', self.share_snapshot.id), ('status', 'available')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.snapshots_mock.reset_state.side_effect = Exception()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_set_snapshot_property_exception(self):
        arglist = ['--property', 'key=', self.share_snapshot.id]
        verifylist = [('property', {'key': ''}), ('snapshot', self.share_snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.share_snapshot.set_metadata.assert_called_with({'key': ''})
        self.share_snapshot.set_metadata.side_effect = exceptions.BadRequest
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)