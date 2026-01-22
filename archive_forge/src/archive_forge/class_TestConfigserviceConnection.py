import os
from tests.unit import unittest
class TestConfigserviceConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.configservice import connect_to_region
        from boto.configservice.layer1 import ConfigServiceConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, ConfigServiceConnection)
        self.assertEqual(connection.host, 'config.us-east-1.amazonaws.com')