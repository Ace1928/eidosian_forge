import os
from tests.unit import unittest
class TestCloudsearchDomainConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.cloudsearchdomain import connect_to_region
        from boto.cloudsearchdomain.layer1 import CloudSearchDomainConnection
        host = 'mycustomdomain.us-east-1.amazonaws.com'
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar', host=host)
        self.assertIsInstance(connection, CloudSearchDomainConnection)
        self.assertEqual(connection.host, host)