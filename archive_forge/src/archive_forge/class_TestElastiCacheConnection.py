import time
from tests.unit import unittest
from boto.elasticache import layer1
from boto.exception import BotoServerError
class TestElastiCacheConnection(unittest.TestCase):

    def setUp(self):
        self.elasticache = layer1.ElastiCacheConnection()

    def wait_until_cluster_available(self, cluster_id):
        timeout = time.time() + 600
        while time.time() < timeout:
            response = self.elasticache.describe_cache_clusters(cluster_id)
            status = response['DescribeCacheClustersResponse']['DescribeCacheClustersResult']['CacheClusters'][0]['CacheClusterStatus']
            if status == 'available':
                break
            time.sleep(5)
        else:
            self.fail('Timeout waiting for cache cluster %rto become available.' % cluster_id)

    def test_create_delete_cache_cluster(self):
        cluster_id = 'cluster-id2'
        self.elasticache.create_cache_cluster(cluster_id, 1, 'cache.t1.micro', 'memcached')
        self.wait_until_cluster_available(cluster_id)
        self.elasticache.delete_cache_cluster(cluster_id)
        timeout = time.time() + 600
        while time.time() < timeout:
            try:
                self.elasticache.describe_cache_clusters(cluster_id)
            except BotoServerError:
                break
            time.sleep(5)
        else:
            self.fail('Timeout waiting for cache cluster %sto be deleted.' % cluster_id)