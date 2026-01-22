from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
class TestVolumeSnapshotList(TestVolumeSnapshot):
    volume = volume_fakes.create_one_volume()
    project = project_fakes.FakeProject.create_one_project()
    snapshots = volume_fakes.create_snapshots(attrs={'volume_id': volume.name}, count=3)
    columns = ['ID', 'Name', 'Description', 'Status', 'Size']
    columns_long = columns + ['Created At', 'Volume', 'Properties']
    data = []
    for s in snapshots:
        data.append((s.id, s.name, s.description, s.status, s.size))
    data_long = []
    for s in snapshots:
        data_long.append((s.id, s.name, s.description, s.status, s.size, s.created_at, volume_snapshot.VolumeIdColumn(s.volume_id, volume_cache={volume.id: volume}), format_columns.DictColumn(s.metadata)))

    def setUp(self):
        super().setUp()
        self.volumes_mock.list.return_value = [self.volume]
        self.volumes_mock.get.return_value = self.volume
        self.project_mock.get.return_value = self.project
        self.snapshots_mock.list.return_value = self.snapshots
        self.cmd = volume_snapshot.ListVolumeSnapshot(self.app, None)

    def test_snapshot_list_without_options(self):
        arglist = []
        verifylist = [('all_projects', False), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.list.assert_called_once_with(limit=None, marker=None, search_opts={'all_tenants': False, 'name': None, 'status': None, 'project_id': None, 'volume_id': None})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_snapshot_list_with_options(self):
        arglist = ['--long', '--limit', '2', '--project', self.project.id, '--marker', self.snapshots[0].id]
        verifylist = [('long', True), ('limit', 2), ('project', self.project.id), ('marker', self.snapshots[0].id), ('all_projects', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.list.assert_called_once_with(limit=2, marker=self.snapshots[0].id, search_opts={'all_tenants': True, 'project_id': self.project.id, 'name': None, 'status': None, 'volume_id': None})
        self.assertEqual(self.columns_long, columns)
        self.assertEqual(self.data_long, list(data))

    def test_snapshot_list_all_projects(self):
        arglist = ['--all-projects']
        verifylist = [('long', False), ('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.list.assert_called_once_with(limit=None, marker=None, search_opts={'all_tenants': True, 'name': None, 'status': None, 'project_id': None, 'volume_id': None})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_snapshot_list_name_option(self):
        arglist = ['--name', self.snapshots[0].name]
        verifylist = [('all_projects', False), ('long', False), ('name', self.snapshots[0].name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.list.assert_called_once_with(limit=None, marker=None, search_opts={'all_tenants': False, 'name': self.snapshots[0].name, 'status': None, 'project_id': None, 'volume_id': None})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_snapshot_list_status_option(self):
        arglist = ['--status', self.snapshots[0].status]
        verifylist = [('all_projects', False), ('long', False), ('status', self.snapshots[0].status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.list.assert_called_once_with(limit=None, marker=None, search_opts={'all_tenants': False, 'name': None, 'status': self.snapshots[0].status, 'project_id': None, 'volume_id': None})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_snapshot_list_volumeid_option(self):
        arglist = ['--volume', self.volume.id]
        verifylist = [('all_projects', False), ('long', False), ('volume', self.volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.list.assert_called_once_with(limit=None, marker=None, search_opts={'all_tenants': False, 'name': None, 'status': None, 'project_id': None, 'volume_id': self.volume.id})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_snapshot_list_negative_limit(self):
        arglist = ['--limit', '-2']
        verifylist = [('limit', -2)]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)