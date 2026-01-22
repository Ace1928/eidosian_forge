import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.native import encoder
from pyasn1.compat.octets import str2octs
from pyasn1.error import PyAsn1Error
class ObjectIdentifierEncoderTestCase(BaseTestCase):

    def testOne(self):
        assert encoder.encode(univ.ObjectIdentifier((1, 3, 6, 0, 12345))) == '1.3.6.0.12345'