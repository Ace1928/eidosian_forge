from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_clusters
from troveclient.tests.osc.v1 import fakes
class TestDatabaseClusterShrink(TestClusters):

    def setUp(self):
        super(TestDatabaseClusterShrink, self).setUp()
        self.cmd = database_clusters.ShrinkDatabaseCluster(self.app, None)
        self.cluster_member = self.fake_clusters.get_clusters_member_2()

    @mock.patch.object(utils, 'find_resource')
    def test_cluster_grow(self, mock_find_resource):
        args = ['test-clstr', 'test-clstr-member-2']
        mock_find_resource.side_effect = [args[0], self.cluster_member]
        parsed_args = self.check_parser(self.cmd, args, [])
        result = self.cmd.take_action(parsed_args)
        self.cluster_client.shrink.assert_called_with('test-clstr', [{'id': 'member-2'}])
        self.assertIsNone(result)