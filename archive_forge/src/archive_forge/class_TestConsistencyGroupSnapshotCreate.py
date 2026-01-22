from unittest.mock import call
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group_snapshot
class TestConsistencyGroupSnapshotCreate(TestConsistencyGroupSnapshot):
    _consistency_group_snapshot = volume_fakes.create_one_consistency_group_snapshot()
    consistency_group = volume_fakes.create_one_consistency_group()
    columns = ('consistencygroup_id', 'created_at', 'description', 'id', 'name', 'status')
    data = (_consistency_group_snapshot.consistencygroup_id, _consistency_group_snapshot.created_at, _consistency_group_snapshot.description, _consistency_group_snapshot.id, _consistency_group_snapshot.name, _consistency_group_snapshot.status)

    def setUp(self):
        super(TestConsistencyGroupSnapshotCreate, self).setUp()
        self.cgsnapshots_mock.create.return_value = self._consistency_group_snapshot
        self.consistencygroups_mock.get.return_value = self.consistency_group
        self.cmd = consistency_group_snapshot.CreateConsistencyGroupSnapshot(self.app, None)

    def test_consistency_group_snapshot_create(self):
        arglist = ['--consistency-group', self.consistency_group.id, '--description', self._consistency_group_snapshot.description, self._consistency_group_snapshot.name]
        verifylist = [('consistency_group', self.consistency_group.id), ('description', self._consistency_group_snapshot.description), ('snapshot_name', self._consistency_group_snapshot.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.consistencygroups_mock.get.assert_called_once_with(self.consistency_group.id)
        self.cgsnapshots_mock.create.assert_called_once_with(self.consistency_group.id, name=self._consistency_group_snapshot.name, description=self._consistency_group_snapshot.description)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_consistency_group_snapshot_create_no_consistency_group(self):
        arglist = ['--description', self._consistency_group_snapshot.description, self._consistency_group_snapshot.name]
        verifylist = [('description', self._consistency_group_snapshot.description), ('snapshot_name', self._consistency_group_snapshot.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.consistencygroups_mock.get.assert_called_once_with(self._consistency_group_snapshot.name)
        self.cgsnapshots_mock.create.assert_called_once_with(self.consistency_group.id, name=self._consistency_group_snapshot.name, description=self._consistency_group_snapshot.description)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)