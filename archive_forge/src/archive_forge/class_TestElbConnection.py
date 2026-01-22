import os
from tests.unit import unittest
class TestElbConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.ec2.elb import connect_to_region
        from boto.ec2.elb import ELBConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, ELBConnection)
        self.assertEqual(connection.host, 'elasticloadbalancing.us-east-1.amazonaws.com')