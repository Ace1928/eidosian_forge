from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_clusters
from troveclient.tests.osc.v1 import fakes
class TestClusterListInstances(TestClusters):
    columns = database_clusters.ListDatabaseClusterInstances.columns
    values = [('member-1', 'test-clstr-member-1', '02', 2, 'ACTIVE'), ('member-2', 'test-clstr-member-2', '2', 2, 'ACTIVE')]

    def setUp(self):
        super(TestClusterListInstances, self).setUp()
        self.cmd = database_clusters.ListDatabaseClusterInstances(self.app, None)
        self.data = self.fake_clusters.get_clusters_cls_1234()
        self.cluster_client.get.return_value = self.data

    def test_cluster_list_instances(self):
        args = ['cls-1234']
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)