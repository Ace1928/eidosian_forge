from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
class TestVolumeSnapshotCreate(TestVolumeSnapshot):
    columns = ('created_at', 'description', 'id', 'name', 'properties', 'size', 'status', 'volume_id')

    def setUp(self):
        super().setUp()
        self.volume = volume_fakes.create_one_volume()
        self.new_snapshot = volume_fakes.create_one_snapshot(attrs={'volume_id': self.volume.id})
        self.data = (self.new_snapshot.created_at, self.new_snapshot.description, self.new_snapshot.id, self.new_snapshot.name, format_columns.DictColumn(self.new_snapshot.metadata), self.new_snapshot.size, self.new_snapshot.status, self.new_snapshot.volume_id)
        self.volumes_mock.get.return_value = self.volume
        self.snapshots_mock.create.return_value = self.new_snapshot
        self.snapshots_mock.manage.return_value = self.new_snapshot
        self.cmd = volume_snapshot.CreateVolumeSnapshot(self.app, None)

    def test_snapshot_create(self):
        arglist = ['--volume', self.new_snapshot.volume_id, '--description', self.new_snapshot.description, '--force', '--property', 'Alpha=a', '--property', 'Beta=b', self.new_snapshot.name]
        verifylist = [('volume', self.new_snapshot.volume_id), ('description', self.new_snapshot.description), ('force', True), ('property', {'Alpha': 'a', 'Beta': 'b'}), ('snapshot_name', self.new_snapshot.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.create.assert_called_with(self.new_snapshot.volume_id, force=True, name=self.new_snapshot.name, description=self.new_snapshot.description, metadata={'Alpha': 'a', 'Beta': 'b'})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_snapshot_create_without_name(self):
        arglist = ['--volume', self.new_snapshot.volume_id]
        verifylist = [('volume', self.new_snapshot.volume_id)]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_snapshot_create_without_volume(self):
        arglist = ['--description', self.new_snapshot.description, '--force', self.new_snapshot.name]
        verifylist = [('description', self.new_snapshot.description), ('force', True), ('snapshot_name', self.new_snapshot.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volumes_mock.get.assert_called_once_with(self.new_snapshot.name)
        self.snapshots_mock.create.assert_called_once_with(self.new_snapshot.volume_id, force=True, name=self.new_snapshot.name, description=self.new_snapshot.description, metadata=None)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_snapshot_create_with_remote_source(self):
        arglist = ['--remote-source', 'source-name=test_source_name', '--remote-source', 'source-id=test_source_id', '--volume', self.new_snapshot.volume_id, self.new_snapshot.name]
        ref_dict = {'source-name': 'test_source_name', 'source-id': 'test_source_id'}
        verifylist = [('remote_source', ref_dict), ('volume', self.new_snapshot.volume_id), ('snapshot_name', self.new_snapshot.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.manage.assert_called_with(volume_id=self.new_snapshot.volume_id, ref=ref_dict, name=self.new_snapshot.name, description=None, metadata=None)
        self.snapshots_mock.create.assert_not_called()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)