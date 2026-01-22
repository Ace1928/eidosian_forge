import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.type import univ
from pyasn1.type import char
from pyasn1.codec.ber import decoder
from pyasn1.codec.ber import eoo
from pyasn1.compat.octets import ints2octs, str2octs, null
from pyasn1.error import PyAsn1Error
class SequenceDecoderWithUnaggedOpenTypesTestCase(BaseTestCase):

    def setUp(self):
        openType = opentype.OpenType('id', {1: univ.Integer(), 2: univ.OctetString()})
        self.s = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.NamedType('id', univ.Integer()), namedtype.NamedType('blob', univ.Any(), openType=openType)))

    def testDecodeOpenTypesChoiceOne(self):
        s, r = decoder.decode(ints2octs((48, 6, 2, 1, 1, 2, 1, 12)), asn1Spec=self.s, decodeOpenTypes=True)
        assert not r
        assert s[0] == 1
        assert s[1] == 12

    def testDecodeOpenTypesChoiceTwo(self):
        s, r = decoder.decode(ints2octs((48, 16, 2, 1, 2, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110)), asn1Spec=self.s, decodeOpenTypes=True)
        assert not r
        assert s[0] == 2
        assert s[1] == univ.OctetString('quick brown')

    def testDecodeOpenTypesUnknownType(self):
        try:
            s, r = decoder.decode(ints2octs((48, 6, 2, 1, 2, 6, 1, 39)), asn1Spec=self.s, decodeOpenTypes=True)
        except PyAsn1Error:
            pass
        else:
            assert False, 'unknown open type tolerated'

    def testDecodeOpenTypesUnknownId(self):
        s, r = decoder.decode(ints2octs((48, 6, 2, 1, 3, 6, 1, 39)), asn1Spec=self.s, decodeOpenTypes=True)
        assert not r
        assert s[0] == 3
        assert s[1] == univ.OctetString(hexValue='060127')

    def testDontDecodeOpenTypesChoiceOne(self):
        s, r = decoder.decode(ints2octs((48, 6, 2, 1, 1, 2, 1, 12)), asn1Spec=self.s)
        assert not r
        assert s[0] == 1
        assert s[1] == ints2octs((2, 1, 12))

    def testDontDecodeOpenTypesChoiceTwo(self):
        s, r = decoder.decode(ints2octs((48, 16, 2, 1, 2, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110)), asn1Spec=self.s)
        assert not r
        assert s[0] == 2
        assert s[1] == ints2octs((4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110))