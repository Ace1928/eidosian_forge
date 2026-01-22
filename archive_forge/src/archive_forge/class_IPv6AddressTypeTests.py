import re
import unittest
from oslo_config import types
class IPv6AddressTypeTests(IPAddressTypeTests):
    type = types.IPAddress(6)

    def test_ipv4_address(self):
        self.assertInvalid('192.168.0.1')