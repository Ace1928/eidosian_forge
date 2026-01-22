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
class UTF8StringEncoderWithSchemaTestCase(BaseTestCase):

    def testEncoding(self):
        assert encoder.encode(sys.version_info[0] == 3 and 'abc' or unicode('abc'), asn1Spec=char.UTF8String()) == ints2octs((12, 3, 97, 98, 99)), 'Incorrect encoding'