import unittest
import time
from nose.plugins.attrib import attr
from boto.redshift.layer1 import RedshiftConnection
from boto.redshift.exceptions import ClusterNotFoundFault
from boto.redshift.exceptions import ResizeNotFoundFault
class TestRedshiftLayer1Management(unittest.TestCase):
    redshift = True

    def setUp(self):
        self.api = RedshiftConnection()
        self.cluster_prefix = 'boto-redshift-cluster-%s'
        self.node_type = 'dw.hs1.xlarge'
        self.master_username = 'mrtest'
        self.master_password = 'P4ssword'
        self.db_name = 'simon'
        self.wait_time = 60 * 20

    def cluster_id(self):
        return self.cluster_prefix % str(int(time.time()))

    def create_cluster(self):
        cluster_id = self.cluster_id()
        self.api.create_cluster(cluster_id, self.node_type, self.master_username, self.master_password, db_name=self.db_name, number_of_nodes=3)
        time.sleep(self.wait_time)
        self.addCleanup(self.delete_cluster_the_slow_way, cluster_id)
        return cluster_id

    def delete_cluster_the_slow_way(self, cluster_id):
        time.sleep(self.wait_time)
        self.api.delete_cluster(cluster_id, skip_final_cluster_snapshot=True)

    @attr('notdefault')
    def test_create_delete_cluster(self):
        cluster_id = self.cluster_id()
        self.api.create_cluster(cluster_id, self.node_type, self.master_username, self.master_password, db_name=self.db_name, number_of_nodes=3)
        time.sleep(self.wait_time)
        self.api.delete_cluster(cluster_id, skip_final_cluster_snapshot=True)

    @attr('notdefault')
    def test_as_much_as_possible_before_teardown(self):
        with self.assertRaises(ClusterNotFoundFault):
            self.api.describe_clusters('badpipelineid')
        cluster_id = self.create_cluster()
        with self.assertRaises(ResizeNotFoundFault):
            self.api.describe_resize(cluster_id)
        clusters = self.api.describe_clusters()['DescribeClustersResponse']['DescribeClustersResult']['Clusters']
        cluster_ids = [c['ClusterIdentifier'] for c in clusters]
        self.assertIn(cluster_id, cluster_ids)
        response = self.api.describe_clusters(cluster_id)
        self.assertEqual(response['DescribeClustersResponse']['DescribeClustersResult']['Clusters'][0]['ClusterIdentifier'], cluster_id)
        snapshot_id = 'snap-%s' % cluster_id
        response = self.api.create_cluster_snapshot(snapshot_id, cluster_id)
        self.assertEqual(response['CreateClusterSnapshotResponse']['CreateClusterSnapshotResult']['Snapshot']['SnapshotIdentifier'], snapshot_id)
        self.assertEqual(response['CreateClusterSnapshotResponse']['CreateClusterSnapshotResult']['Snapshot']['Status'], 'creating')
        self.addCleanup(self.api.delete_cluster_snapshot, snapshot_id)
        time.sleep(self.wait_time)
        response = self.api.describe_cluster_snapshots(cluster_identifier=cluster_id)
        snap = response['DescribeClusterSnapshotsResponse']['DescribeClusterSnapshotsResult']['Snapshots'][-1]
        self.assertEqual(snap['SnapshotType'], 'manual')
        self.assertEqual(snap['DBName'], self.db_name)