import os
from tests.unit import unittest
class TestEC2Connection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.ec2 import connect_to_region
        from boto.ec2.connection import EC2Connection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, EC2Connection)
        self.assertEqual(connection.host, 'ec2.us-east-1.amazonaws.com')