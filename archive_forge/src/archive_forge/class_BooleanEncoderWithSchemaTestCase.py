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
class BooleanEncoderWithSchemaTestCase(BaseTestCase):

    def testTrue(self):
        assert encoder.encode(True, asn1Spec=univ.Boolean()) == ints2octs((1, 1, 1))

    def testFalse(self):
        assert encoder.encode(False, asn1Spec=univ.Boolean()) == ints2octs((1, 1, 0))