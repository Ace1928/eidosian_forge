from unittest import mock
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_migration
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestListMigration(TestServerMigration):
    """Test fetch all migrations."""
    MIGRATION_COLUMNS = ['Source Node', 'Dest Node', 'Source Compute', 'Dest Compute', 'Dest Host', 'Status', 'Server UUID', 'Old Flavor', 'New Flavor', 'Created At', 'Updated At']
    MIGRATION_FIELDS = ['source_node', 'dest_node', 'source_compute', 'dest_compute', 'dest_host', 'status', 'server_id', 'old_flavor_id', 'new_flavor_id', 'created_at', 'updated_at']

    def setUp(self):
        super().setUp()
        self._set_mock_microversion('2.1')
        self.server = compute_fakes.create_one_sdk_server()
        self.compute_sdk_client.find_server.return_value = self.server
        self.migrations = compute_fakes.create_migrations(count=3)
        self.compute_sdk_client.migrations.return_value = self.migrations
        self.data = (common_utils.get_item_properties(s, self.MIGRATION_FIELDS) for s in self.migrations)
        self.cmd = server_migration.ListMigration(self.app, None)

    def test_server_migration_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {}
        self.compute_sdk_client.migrations.assert_called_with(**kwargs)
        self.assertEqual(self.MIGRATION_COLUMNS, columns)
        self.assertEqual(tuple(self.data), tuple(data))

    def test_server_migration_list(self):
        arglist = ['--server', 'server1', '--host', 'host1', '--status', 'migrating', '--type', 'cold-migration']
        verifylist = [('server', 'server1'), ('host', 'host1'), ('status', 'migrating')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'status': 'migrating', 'host': 'host1', 'instance_uuid': self.server.id, 'migration_type': 'migration'}
        self.compute_sdk_client.find_server.assert_called_with('server1', ignore_missing=False)
        self.compute_sdk_client.migrations.assert_called_with(**kwargs)
        self.assertEqual(self.MIGRATION_COLUMNS, columns)
        self.assertEqual(tuple(self.data), tuple(data))