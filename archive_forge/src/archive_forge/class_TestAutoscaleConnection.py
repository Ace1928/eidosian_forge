import os
from tests.unit import unittest
class TestAutoscaleConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.ec2.autoscale import connect_to_region
        from boto.ec2.autoscale import AutoScaleConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, AutoScaleConnection)
        self.assertEqual(connection.host, 'autoscaling.us-east-1.amazonaws.com')