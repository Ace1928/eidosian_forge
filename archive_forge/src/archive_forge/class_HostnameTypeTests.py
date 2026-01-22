import re
import unittest
from oslo_config import types
class HostnameTypeTests(TypeTestHelper, unittest.TestCase):
    type = types.Hostname()

    def assertConvertedEqual(self, value):
        self.assertConvertedValue(value, value)

    def test_empty_hostname_fails(self):
        self.assertInvalid('')

    def test_should_return_same_hostname_if_valid(self):
        self.assertConvertedEqual('foo.bar')

    def test_trailing_quote_is_invalid(self):
        self.assertInvalid('foo.bar"')

    def test_repr(self):
        self.assertEqual('Hostname', repr(types.Hostname()))

    def test_equal(self):
        self.assertEqual(types.Hostname(), types.Hostname())

    def test_not_equal_to_other_class(self):
        self.assertNotEqual(types.Hostname(), types.Integer())
        self.assertNotEqual(types.Hostname(), types.String())

    def test_invalid_characters(self):
        self.assertInvalid('"host"')
        self.assertInvalid("h'ost'")
        self.assertInvalid("h'ost")
        self.assertInvalid('h$ost')
        self.assertInvalid('host_01.co.uk')
        self.assertInvalid('h%ost')
        self.assertInvalid('host;name=99')
        self.assertInvalid('___site0.1001')
        self.assertInvalid('_site01001')
        self.assertInvalid('host..name')
        self.assertInvalid('.host.name.com')
        self.assertInvalid('no spaces')

    def test_invalid_hostnames_with_numeric_characters(self):
        self.assertInvalid('10.0.0.0')
        self.assertInvalid('3.14')
        self.assertInvalid('___site0.1001')
        self.assertInvalid('org.10')
        self.assertInvalid('0.0.00')

    def test_no_start_end_hyphens(self):
        self.assertInvalid('-host.com')
        self.assertInvalid('-hostname.com-')
        self.assertInvalid('hostname.co.uk-')

    def test_strip_trailing_dot(self):
        self.assertConvertedValue('cell1.nova.site1.', 'cell1.nova.site1')
        self.assertConvertedValue('cell1.', 'cell1')

    def test_valid_hostname(self):
        self.assertConvertedEqual('cell1.nova.site1')
        self.assertConvertedEqual('site01001')
        self.assertConvertedEqual('home-site-here.org.com')
        self.assertConvertedEqual('localhost')
        self.assertConvertedEqual('3com.com')
        self.assertConvertedEqual('10.org')
        self.assertConvertedEqual('10ab.10ab')
        self.assertConvertedEqual('ab-c.com')
        self.assertConvertedEqual('abc.com-org')
        self.assertConvertedEqual('abc.0-0')

    def test_max_segment_size(self):
        self.assertConvertedEqual('host.%s.com' % ('x' * 63))
        self.assertInvalid('host.%s.com' % ('x' * 64))

    def test_max_hostname_size(self):
        test_str = '.'.join(('x' * 31 for x in range(8)))
        self.assertEqual(255, len(test_str))
        self.assertInvalid(test_str)
        self.assertConvertedEqual(test_str[:-2])