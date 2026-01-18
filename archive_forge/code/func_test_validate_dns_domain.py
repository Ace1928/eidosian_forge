from unittest import mock
from neutron_lib.api.validators import dns
from neutron_lib.db import constants as db_constants
from neutron_lib.tests import _base as base
def test_validate_dns_domain(self):
    msg = dns.validate_dns_domain('')
    self.assertIsNone(msg)
    msg = dns.validate_dns_domain('example.com.')
    self.assertIsNone(msg)
    invalid_data = 1234
    expected = "'%s' is not a valid string" % invalid_data
    msg = dns.validate_dns_domain(invalid_data)
    self.assertEqual(expected, msg)
    invalid_data = 'example.com'
    expected = "'%s' is not a FQDN" % invalid_data
    msg = dns.validate_dns_domain(invalid_data)
    self.assertEqual(expected, msg)
    length = 9
    invalid_data = 'a' * length + '.'
    max_len = 11
    expected = "'%(data)s' contains %(length)s characters. Adding a sub-domain will cause it to exceed the maximum length of a FQDN of '%(max_len)s'" % {'data': invalid_data, 'length': length + 1, 'max_len': max_len}
    msg = dns.validate_dns_domain(invalid_data, max_len)
    self.assertEqual(expected, msg)