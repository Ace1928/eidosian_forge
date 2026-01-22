import os
from tests.unit import unittest
class TestCloudsearchConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.cloudsearch import connect_to_region
        from boto.cloudsearch.layer1 import Layer1
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, Layer1)
        self.assertEqual(connection.host, 'cloudsearch.us-east-1.amazonaws.com')