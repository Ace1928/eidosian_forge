from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
def test_snapshot_list_with_options(self):
    arglist = ['--long', '--limit', '2', '--project', self.project.id, '--marker', self.snapshots[0].id]
    verifylist = [('long', True), ('limit', 2), ('project', self.project.id), ('marker', self.snapshots[0].id), ('all_projects', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.snapshots_mock.list.assert_called_once_with(limit=2, marker=self.snapshots[0].id, search_opts={'all_tenants': True, 'project_id': self.project.id, 'name': None, 'status': None, 'volume_id': None})
    self.assertEqual(self.columns_long, columns)
    self.assertEqual(self.data_long, list(data))