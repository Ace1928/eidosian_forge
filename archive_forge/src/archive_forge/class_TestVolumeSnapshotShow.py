from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
class TestVolumeSnapshotShow(TestVolumeSnapshot):
    columns = ('created_at', 'description', 'id', 'name', 'properties', 'size', 'status', 'volume_id')

    def setUp(self):
        super().setUp()
        self.snapshot = volume_fakes.create_one_snapshot()
        self.data = (self.snapshot.created_at, self.snapshot.description, self.snapshot.id, self.snapshot.name, format_columns.DictColumn(self.snapshot.metadata), self.snapshot.size, self.snapshot.status, self.snapshot.volume_id)
        self.snapshots_mock.get.return_value = self.snapshot
        self.cmd = volume_snapshot.ShowVolumeSnapshot(self.app, None)

    def test_snapshot_show(self):
        arglist = [self.snapshot.id]
        verifylist = [('snapshot', self.snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.get.assert_called_with(self.snapshot.id)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)