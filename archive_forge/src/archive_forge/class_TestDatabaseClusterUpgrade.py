from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_clusters
from troveclient.tests.osc.v1 import fakes
class TestDatabaseClusterUpgrade(TestClusters):

    def setUp(self):
        super(TestDatabaseClusterUpgrade, self).setUp()
        self.cmd = database_clusters.UpgradeDatabaseCluster(self.app, None)

    @mock.patch.object(utils, 'find_resource')
    def test_cluster_upgrade(self, mock_find):
        args = ['cluster1', 'datastore_version1']
        mock_find.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        result = self.cmd.take_action(parsed_args)
        self.cluster_client.upgrade.assert_called_with('cluster1', 'datastore_version1')
        self.assertIsNone(result)