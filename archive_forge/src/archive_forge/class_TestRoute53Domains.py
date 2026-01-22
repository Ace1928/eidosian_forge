import boto
from boto.route53.domains.exceptions import InvalidInput
from tests.compat import unittest
class TestRoute53Domains(unittest.TestCase):

    def setUp(self):
        self.route53domains = boto.connect_route53domains()

    def test_check_domain_availability(self):
        response = self.route53domains.check_domain_availability(domain_name='amazon.com', idn_lang_code='eng')
        self.assertEqual(response, {'Availability': 'UNAVAILABLE'})

    def test_handle_invalid_input_error(self):
        with self.assertRaises(InvalidInput):
            self.route53domains.check_domain_availability(domain_name='!amazon.com')