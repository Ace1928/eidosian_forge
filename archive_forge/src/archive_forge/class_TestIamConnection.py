import os
from tests.unit import unittest
class TestIamConnection(unittest.TestCase):

    def assert_connection(self, region, host):
        from boto.iam import connect_to_region
        from boto.iam.connection import IAMConnection
        connection = connect_to_region(region, aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, IAMConnection)
        self.assertEqual(connection.host, host)

    def test_connect_to_region(self):
        self.assert_connection('us-east-1', 'iam.amazonaws.com')

    def test_connect_to_universal_region(self):
        self.assert_connection('universal', 'iam.amazonaws.com')