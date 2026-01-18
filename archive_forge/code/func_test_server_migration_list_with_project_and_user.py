from unittest import mock
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_migration
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_server_migration_list_with_project_and_user(self):
    arglist = ['--status', 'migrating', '--limit', '1', '--changes-since', '2019-08-07T08:03:25Z', '--changes-before', '2019-08-09T08:03:25Z', '--project', self.project.id, '--user', self.user.id]
    verifylist = [('status', 'migrating'), ('limit', 1), ('changes_since', '2019-08-07T08:03:25Z'), ('changes_before', '2019-08-09T08:03:25Z'), ('project', self.project.id), ('user', self.user.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'status': 'migrating', 'limit': 1, 'paginated': False, 'project_id': self.project.id, 'user_id': self.user.id, 'changes_since': '2019-08-07T08:03:25Z', 'changes_before': '2019-08-09T08:03:25Z'}
    self.compute_sdk_client.migrations.assert_called_with(**kwargs)
    self.MIGRATION_COLUMNS.insert(len(self.MIGRATION_COLUMNS) - 2, 'Project')
    self.MIGRATION_FIELDS.insert(len(self.MIGRATION_FIELDS) - 2, 'project_id')
    self.MIGRATION_COLUMNS.insert(len(self.MIGRATION_COLUMNS) - 2, 'User')
    self.MIGRATION_FIELDS.insert(len(self.MIGRATION_FIELDS) - 2, 'user_id')
    self.assertEqual(self.MIGRATION_COLUMNS, columns)
    self.assertEqual(tuple(self.data), tuple(data))
    self.MIGRATION_COLUMNS.remove('Project')
    self.MIGRATION_FIELDS.remove('project_id')
    self.MIGRATION_COLUMNS.remove('User')
    self.MIGRATION_FIELDS.remove('user_id')