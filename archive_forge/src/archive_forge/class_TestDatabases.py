from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import databases
from troveclient.tests.osc.v1 import fakes
class TestDatabases(fakes.TestDatabasev1):
    fake_databases = fakes.FakeDatabases()

    def setUp(self):
        super(TestDatabases, self).setUp()
        self.mock_client = self.app.client_manager.database
        self.database_client = self.app.client_manager.database.databases