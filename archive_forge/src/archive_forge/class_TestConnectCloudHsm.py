import os
from tests.unit import unittest
class TestConnectCloudHsm(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.cloudhsm import connect_to_region
        from boto.cloudhsm.layer1 import CloudHSMConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, CloudHSMConnection)
        self.assertEqual(connection.host, 'cloudhsm.us-east-1.amazonaws.com')