import os
from tests.unit import unittest
class TestCloudTrailConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.cloudtrail import connect_to_region
        from boto.cloudtrail.layer1 import CloudTrailConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, CloudTrailConnection)
        self.assertEqual(connection.host, 'cloudtrail.us-east-1.amazonaws.com')