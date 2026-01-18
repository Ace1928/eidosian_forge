from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
def test_snapshot_list_all_projects(self):
    arglist = ['--all-projects']
    verifylist = [('long', False), ('all_projects', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.snapshots_mock.list.assert_called_once_with(limit=None, marker=None, search_opts={'all_tenants': True, 'name': None, 'status': None, 'project_id': None, 'volume_id': None})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))