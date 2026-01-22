import boto
from tests.compat import unittest
class DirectConnectTest(unittest.TestCase):
    """
    A very basic test to make sure signatures and
    basic calls work.
    """

    def test_basic(self):
        conn = boto.connect_directconnect()
        response = conn.describe_connections()
        self.assertTrue(response)
        self.assertTrue('connections' in response)
        self.assertIsInstance(response['connections'], list)