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
class AnyEncoderWithSchemaTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.Any()
        self.v = encoder.encode(univ.OctetString('fox'))

    def testUntagged(self):
        assert encoder.encode(self.v, asn1Spec=self.s) == ints2octs((4, 3, 102, 111, 120))

    def testTaggedEx(self):
        s = self.s.subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 4))
        assert encoder.encode(self.v, asn1Spec=s) == ints2octs((164, 5, 4, 3, 102, 111, 120))

    def testTaggedIm(self):
        s = self.s.subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 4))
        assert encoder.encode(self.v, asn1Spec=s) == ints2octs((132, 5, 4, 3, 102, 111, 120))