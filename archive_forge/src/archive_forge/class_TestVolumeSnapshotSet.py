from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
class TestVolumeSnapshotSet(TestVolumeSnapshot):
    snapshot = volume_fakes.create_one_snapshot()

    def setUp(self):
        super().setUp()
        self.snapshots_mock.get.return_value = self.snapshot
        self.snapshots_mock.set_metadata.return_value = None
        self.snapshots_mock.update.return_value = None
        self.cmd = volume_snapshot.SetVolumeSnapshot(self.app, None)

    def test_snapshot_set_no_option(self):
        arglist = [self.snapshot.id]
        verifylist = [('snapshot', self.snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.snapshots_mock.get.assert_called_once_with(parsed_args.snapshot)
        self.assertNotCalled(self.snapshots_mock.reset_state)
        self.assertNotCalled(self.snapshots_mock.update)
        self.assertNotCalled(self.snapshots_mock.set_metadata)
        self.assertIsNone(result)

    def test_snapshot_set_name_and_property(self):
        arglist = ['--name', 'new_snapshot', '--property', 'x=y', '--property', 'foo=foo', self.snapshot.id]
        new_property = {'x': 'y', 'foo': 'foo'}
        verifylist = [('name', 'new_snapshot'), ('property', new_property), ('snapshot', self.snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'name': 'new_snapshot'}
        self.snapshots_mock.update.assert_called_with(self.snapshot.id, **kwargs)
        self.snapshots_mock.set_metadata.assert_called_with(self.snapshot.id, new_property)
        self.assertIsNone(result)

    def test_snapshot_set_with_no_property(self):
        arglist = ['--no-property', self.snapshot.id]
        verifylist = [('no_property', True), ('snapshot', self.snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.snapshots_mock.get.assert_called_once_with(parsed_args.snapshot)
        self.assertNotCalled(self.snapshots_mock.reset_state)
        self.assertNotCalled(self.snapshots_mock.update)
        self.assertNotCalled(self.snapshots_mock.set_metadata)
        self.snapshots_mock.delete_metadata.assert_called_with(self.snapshot.id, ['foo'])
        self.assertIsNone(result)

    def test_snapshot_set_with_no_property_and_property(self):
        arglist = ['--no-property', '--property', 'foo_1=bar_1', self.snapshot.id]
        verifylist = [('no_property', True), ('property', {'foo_1': 'bar_1'}), ('snapshot', self.snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.snapshots_mock.get.assert_called_once_with(parsed_args.snapshot)
        self.assertNotCalled(self.snapshots_mock.reset_state)
        self.assertNotCalled(self.snapshots_mock.update)
        self.snapshots_mock.delete_metadata.assert_called_with(self.snapshot.id, ['foo'])
        self.snapshots_mock.set_metadata.assert_called_once_with(self.snapshot.id, {'foo_1': 'bar_1'})
        self.assertIsNone(result)

    def test_snapshot_set_state_to_error(self):
        arglist = ['--state', 'error', self.snapshot.id]
        verifylist = [('state', 'error'), ('snapshot', self.snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.snapshots_mock.reset_state.assert_called_with(self.snapshot.id, 'error')
        self.assertIsNone(result)

    def test_volume_set_state_failed(self):
        self.snapshots_mock.reset_state.side_effect = exceptions.CommandError()
        arglist = ['--state', 'error', self.snapshot.id]
        verifylist = [('state', 'error'), ('snapshot', self.snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('One or more of the set operations failed', str(e))
        self.snapshots_mock.reset_state.assert_called_once_with(self.snapshot.id, 'error')

    def test_volume_set_name_and_state_failed(self):
        self.snapshots_mock.reset_state.side_effect = exceptions.CommandError()
        arglist = ['--state', 'error', '--name', 'new_snapshot', self.snapshot.id]
        verifylist = [('state', 'error'), ('name', 'new_snapshot'), ('snapshot', self.snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('One or more of the set operations failed', str(e))
        kwargs = {'name': 'new_snapshot'}
        self.snapshots_mock.update.assert_called_once_with(self.snapshot.id, **kwargs)
        self.snapshots_mock.reset_state.assert_called_once_with(self.snapshot.id, 'error')