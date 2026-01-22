import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class BitStringEncoderTestCase(BaseTestCase):

    def testShortMode(self):
        assert encoder.encode(univ.BitString((1, 0) * 5)) == ints2octs((3, 3, 6, 170, 128))

    def testLongMode(self):
        assert encoder.encode(univ.BitString((1, 0) * 501)) == ints2octs((3, 127, 6) + (170,) * 125 + (128,))