import os
from tests.unit import unittest
class TestSesConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.ses import connect_to_region
        from boto.ses.connection import SESConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, SESConnection)
        self.assertEqual(connection.host, 'email.us-east-1.amazonaws.com')