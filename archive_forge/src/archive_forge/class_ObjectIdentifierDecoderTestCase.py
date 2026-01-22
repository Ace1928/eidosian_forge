import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.native import decoder
from pyasn1.error import PyAsn1Error
class ObjectIdentifierDecoderTestCase(BaseTestCase):

    def testOne(self):
        assert decoder.decode('1.3.6.11', asn1Spec=univ.ObjectIdentifier()) == univ.ObjectIdentifier('1.3.6.11')