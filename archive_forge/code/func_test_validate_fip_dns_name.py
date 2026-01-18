from unittest import mock
from neutron_lib.api.validators import dns
from neutron_lib.db import constants as db_constants
from neutron_lib.tests import _base as base
def test_validate_fip_dns_name(self):
    msg = dns.validate_fip_dns_name('')
    self.assertIsNone(msg)
    msg = dns.validate_fip_dns_name('host')
    self.assertIsNone(msg)
    invalid_data = 1234
    expected = "'%s' is not a valid string" % invalid_data
    msg = dns.validate_fip_dns_name(invalid_data)
    self.assertEqual(expected, msg)
    invalid_data = 'host.'
    expected = "'%s' is a FQDN. It should be a relative domain name" % invalid_data
    msg = dns.validate_fip_dns_name(invalid_data)
    self.assertEqual(expected, msg)
    length = 10
    invalid_data = 'a' * length
    max_len = 12
    expected = "'%(data)s' contains %(length)s characters. Adding a domain name will cause it to exceed the maximum length of a FQDN of '%(max_len)s'" % {'data': invalid_data, 'length': length, 'max_len': max_len}
    msg = dns.validate_fip_dns_name(invalid_data, max_len)
    self.assertEqual(expected, msg)