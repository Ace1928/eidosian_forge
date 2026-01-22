import sys
from tests.base import BaseTestCase
from pyasn1.codec.cer import decoder
from pyasn1.compat.octets import ints2octs, str2octs, null
from pyasn1.error import PyAsn1Error
class BitStringDecoderTestCase(BaseTestCase):

    def testShortMode(self):
        assert decoder.decode(ints2octs((3, 3, 6, 170, 128))) == ((1, 0) * 5, null)

    def testLongMode(self):
        assert decoder.decode(ints2octs((3, 127, 6) + (170,) * 125 + (128,))) == ((1, 0) * 501, null)