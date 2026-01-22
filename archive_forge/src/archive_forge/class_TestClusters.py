from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_clusters
from troveclient.tests.osc.v1 import fakes
class TestClusters(fakes.TestDatabasev1):
    fake_clusters = fakes.FakeClusters()

    def setUp(self):
        super(TestClusters, self).setUp()
        self.mock_client = self.app.client_manager.database
        self.cluster_client = self.app.client_manager.database.clusters
        self.instance_client = self.app.client_manager.database.instances