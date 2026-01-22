import os
from tests.unit import unittest
class TestOpsworksConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.opsworks import connect_to_region
        from boto.opsworks.layer1 import OpsWorksConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, OpsWorksConnection)
        self.assertEqual(connection.host, 'opsworks.us-east-1.amazonaws.com')