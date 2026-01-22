import os
from tests.unit import unittest
class TestVpcConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.vpc import connect_to_region
        from boto.vpc import VPCConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, VPCConnection)
        self.assertEqual(connection.host, 'ec2.us-east-1.amazonaws.com')