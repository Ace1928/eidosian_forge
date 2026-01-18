import sys
from tests.base import BaseTestCase
from pyasn1.compat import octets
def test_str2octs(self):
    assert '\x01\x02\x03' == octets.str2octs('\x01\x02\x03')