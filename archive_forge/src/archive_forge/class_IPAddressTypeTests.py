import re
import unittest
from oslo_config import types
class IPAddressTypeTests(TypeTestHelper, unittest.TestCase):
    type = types.IPAddress()

    def test_ipv4_address(self):
        self.assertConvertedValue('192.168.0.1', '192.168.0.1')

    def test_ipv6_address(self):
        self.assertConvertedValue('abcd:ef::1', 'abcd:ef::1')

    def test_strings(self):
        self.assertInvalid('')
        self.assertInvalid('foo')

    def test_numbers(self):
        self.assertInvalid(1)
        self.assertInvalid(-1)
        self.assertInvalid(3.14)