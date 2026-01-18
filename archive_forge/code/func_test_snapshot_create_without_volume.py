from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
def test_snapshot_create_without_volume(self):
    arglist = ['--description', self.new_snapshot.description, '--force', self.new_snapshot.name]
    verifylist = [('description', self.new_snapshot.description), ('force', True), ('snapshot_name', self.new_snapshot.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volumes_mock.get.assert_called_once_with(self.new_snapshot.name)
    self.snapshots_mock.create.assert_called_once_with(self.new_snapshot.volume_id, force=True, name=self.new_snapshot.name, description=self.new_snapshot.description, metadata=None)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)