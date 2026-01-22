import os
from tests.unit import unittest
class TestRoute53DomainsConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.route53.domains import connect_to_region
        from boto.route53.domains.layer1 import Route53DomainsConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, Route53DomainsConnection)
        self.assertEqual(connection.host, 'route53domains.us-east-1.amazonaws.com')