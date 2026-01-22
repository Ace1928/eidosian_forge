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
class UniversalStringEncoderWithSchemaTestCase(BaseTestCase):

    def testEncoding(self):
        assert encoder.encode(sys.version_info[0] == 3 and 'abc' or unicode('abc'), asn1Spec=char.UniversalString()) == ints2octs((28, 12, 0, 0, 0, 97, 0, 0, 0, 98, 0, 0, 0, 99)), 'Incorrect encoding'