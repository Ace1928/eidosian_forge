import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def test_sizedIntegerTypes(self):
    """
        Test that integers below the maximum C{INT} token size cutoff are
        serialized as C{INT} or C{NEG} and that larger integers are
        serialized as C{LONGINT} or C{LONGNEG}.
        """
    baseIntIn = +2147483647
    baseNegIn = -2147483648
    baseIntOut = b'\x7f\x7f\x7f\x07\x81'
    self.assertEqual(self.encode(baseIntIn - 2), b'}' + baseIntOut)
    self.assertEqual(self.encode(baseIntIn - 1), b'~' + baseIntOut)
    self.assertEqual(self.encode(baseIntIn - 0), b'\x7f' + baseIntOut)
    baseLongIntOut = b'\x00\x00\x00\x08\x85'
    self.assertEqual(self.encode(baseIntIn + 1), b'\x00' + baseLongIntOut)
    self.assertEqual(self.encode(baseIntIn + 2), b'\x01' + baseLongIntOut)
    self.assertEqual(self.encode(baseIntIn + 3), b'\x02' + baseLongIntOut)
    baseNegOut = b'\x7f\x7f\x7f\x07\x83'
    self.assertEqual(self.encode(baseNegIn + 2), b'~' + baseNegOut)
    self.assertEqual(self.encode(baseNegIn + 1), b'\x7f' + baseNegOut)
    self.assertEqual(self.encode(baseNegIn + 0), b'\x00\x00\x00\x00\x08\x83')
    baseLongNegOut = b'\x00\x00\x00\x08\x86'
    self.assertEqual(self.encode(baseNegIn - 1), b'\x01' + baseLongNegOut)
    self.assertEqual(self.encode(baseNegIn - 2), b'\x02' + baseLongNegOut)
    self.assertEqual(self.encode(baseNegIn - 3), b'\x03' + baseLongNegOut)