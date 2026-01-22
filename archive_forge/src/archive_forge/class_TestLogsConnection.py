import os
from tests.unit import unittest
class TestLogsConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.logs import connect_to_region
        from boto.logs.layer1 import CloudWatchLogsConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, CloudWatchLogsConnection)
        self.assertEqual(connection.host, 'logs.us-east-1.amazonaws.com')