import os
from tests.unit import unittest
class TestElastictranscoderConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.elastictranscoder import connect_to_region
        from boto.elastictranscoder.layer1 import ElasticTranscoderConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, ElasticTranscoderConnection)
        self.assertEqual(connection.host, 'elastictranscoder.us-east-1.amazonaws.com')