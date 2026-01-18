import sys
from tests.base import BaseTestCase
from pyasn1.compat import octets
def test_octs2ints(self):
    assert [1, 2, 3] == octets.octs2ints('\x01\x02\x03')