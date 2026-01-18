from unittest import mock
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_migration
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_server_migration_abort_by_uuid(self):
    self._set_mock_microversion('2.59')
    self.server_migration = compute_fakes.create_one_server_migration()
    self.compute_sdk_client.server_migrations.return_value = iter([self.server_migration])
    arglist = [self.server.id, self.server_migration.uuid]
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.find_server.assert_called_with(self.server.id, ignore_missing=False)
    self.compute_sdk_client.server_migrations.assert_called_with(self.server.id)
    self.compute_sdk_client.abort_server_migration.assert_called_with(self.server_migration.id, self.server.id, ignore_missing=False)
    self.assertIsNone(result)