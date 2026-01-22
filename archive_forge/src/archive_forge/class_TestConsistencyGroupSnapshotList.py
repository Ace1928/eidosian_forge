from unittest.mock import call
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group_snapshot
class TestConsistencyGroupSnapshotList(TestConsistencyGroupSnapshot):
    consistency_group_snapshots = volume_fakes.create_consistency_group_snapshots(count=2)
    consistency_group = volume_fakes.create_one_consistency_group()
    columns = ['ID', 'Status', 'Name']
    columns_long = ['ID', 'Status', 'ConsistencyGroup ID', 'Name', 'Description', 'Created At']
    data = []
    for c in consistency_group_snapshots:
        data.append((c.id, c.status, c.name))
    data_long = []
    for c in consistency_group_snapshots:
        data_long.append((c.id, c.status, c.consistencygroup_id, c.name, c.description, c.created_at))

    def setUp(self):
        super(TestConsistencyGroupSnapshotList, self).setUp()
        self.cgsnapshots_mock.list.return_value = self.consistency_group_snapshots
        self.consistencygroups_mock.get.return_value = self.consistency_group
        self.cmd = consistency_group_snapshot.ListConsistencyGroupSnapshot(self.app, None)

    def test_consistency_group_snapshot_list_without_options(self):
        arglist = []
        verifylist = [('all_projects', False), ('long', False), ('status', None), ('consistency_group', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': False, 'status': None, 'consistencygroup_id': None}
        self.cgsnapshots_mock.list.assert_called_once_with(detailed=True, search_opts=search_opts)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_consistency_group_snapshot_list_with_long(self):
        arglist = ['--long']
        verifylist = [('all_projects', False), ('long', True), ('status', None), ('consistency_group', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': False, 'status': None, 'consistencygroup_id': None}
        self.cgsnapshots_mock.list.assert_called_once_with(detailed=True, search_opts=search_opts)
        self.assertEqual(self.columns_long, columns)
        self.assertEqual(self.data_long, list(data))

    def test_consistency_group_snapshot_list_with_options(self):
        arglist = ['--all-project', '--status', self.consistency_group_snapshots[0].status, '--consistency-group', self.consistency_group.id]
        verifylist = [('all_projects', True), ('long', False), ('status', self.consistency_group_snapshots[0].status), ('consistency_group', self.consistency_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': True, 'status': self.consistency_group_snapshots[0].status, 'consistencygroup_id': self.consistency_group.id}
        self.consistencygroups_mock.get.assert_called_once_with(self.consistency_group.id)
        self.cgsnapshots_mock.list.assert_called_once_with(detailed=True, search_opts=search_opts)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))