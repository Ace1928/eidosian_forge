from novaclient import api_versions
from novaclient import base
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import server_migrations as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import server_migrations
class ServerMigrationsTestV224(ServerMigrationsTest):

    def setUp(self):
        super(ServerMigrationsTestV224, self).setUp()
        self.cs.api_version = api_versions.APIVersion('2.24')

    def test_live_migration_abort(self):
        self.cs.server_migrations.live_migration_abort(1234, 1)
        self.assert_called('DELETE', '/servers/1234/migrations/1')