import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.type import univ
from pyasn1.type import char
from pyasn1.codec.ber import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class RealEncoderWithSchemaTestCase(BaseTestCase):

    def testChar(self):
        assert encoder.encode((123, 10, 11), asn1Spec=univ.Real()) == ints2octs((9, 7, 3, 49, 50, 51, 69, 49, 49))