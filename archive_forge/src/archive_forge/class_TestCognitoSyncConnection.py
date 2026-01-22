import os
from tests.unit import unittest
class TestCognitoSyncConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.cognito.sync import connect_to_region
        from boto.cognito.sync.layer1 import CognitoSyncConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, CognitoSyncConnection)
        self.assertEqual(connection.host, 'cognito-sync.us-east-1.amazonaws.com')