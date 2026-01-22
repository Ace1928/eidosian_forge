from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
class TestConsistencyGroupShow(TestConsistencyGroup):
    columns = ('availability_zone', 'created_at', 'description', 'id', 'name', 'status', 'volume_types')

    def setUp(self):
        super().setUp()
        self.consistency_group = volume_fakes.create_one_consistency_group()
        self.data = (self.consistency_group.availability_zone, self.consistency_group.created_at, self.consistency_group.description, self.consistency_group.id, self.consistency_group.name, self.consistency_group.status, self.consistency_group.volume_types)
        self.consistencygroups_mock.get.return_value = self.consistency_group
        self.cmd = consistency_group.ShowConsistencyGroup(self.app, None)

    def test_consistency_group_show(self):
        arglist = [self.consistency_group.id]
        verifylist = [('consistency_group', self.consistency_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.consistencygroups_mock.get.assert_called_once_with(self.consistency_group.id)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)