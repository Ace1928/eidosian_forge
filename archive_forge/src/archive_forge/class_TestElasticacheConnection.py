import os
from tests.unit import unittest
class TestElasticacheConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.elasticache import connect_to_region
        from boto.elasticache.layer1 import ElastiCacheConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, ElastiCacheConnection)
        self.assertEqual(connection.host, 'elasticache.us-east-1.amazonaws.com')