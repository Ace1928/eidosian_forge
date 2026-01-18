import sys
from tests.base import BaseTestCase
from pyasn1.compat import octets
def test_octs2ints_empty(self):
    assert not octets.octs2ints('')