import os
from tests.unit import unittest
class TestRds2Connection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.rds2 import connect_to_region
        from boto.rds2.layer1 import RDSConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, RDSConnection)
        self.assertEqual(connection.host, 'rds.amazonaws.com')