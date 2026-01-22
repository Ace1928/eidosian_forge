import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.native import decoder
from pyasn1.error import PyAsn1Error
class RealDecoderTestCase(BaseTestCase):

    def testSimple(self):
        assert decoder.decode(1.33, asn1Spec=univ.Real()) == univ.Real(1.33)