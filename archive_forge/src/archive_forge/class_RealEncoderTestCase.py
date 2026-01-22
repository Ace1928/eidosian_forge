import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.native import encoder
from pyasn1.compat.octets import str2octs
from pyasn1.error import PyAsn1Error
class RealEncoderTestCase(BaseTestCase):

    def testChar(self):
        assert encoder.encode(univ.Real((123, 10, 11))) == 12300000000000.0

    def testPlusInf(self):
        assert encoder.encode(univ.Real('inf')) == float('inf')

    def testMinusInf(self):
        assert encoder.encode(univ.Real('-inf')) == float('-inf')