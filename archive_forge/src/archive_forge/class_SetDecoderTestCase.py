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
class SetDecoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.Set(componentType=namedtype.NamedTypes(namedtype.NamedType('place-holder', univ.Null(null)), namedtype.NamedType('first-name', univ.OctetString(null)), namedtype.NamedType('age', univ.Integer(33))))
        self.s.setComponentByPosition(0, univ.Null(null))
        self.s.setComponentByPosition(1, univ.OctetString('quick brown'))
        self.s.setComponentByPosition(2, univ.Integer(1))

    def testWithOptionalAndDefaultedDefMode(self):
        assert decoder.decode(ints2octs((49, 18, 5, 0, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 2, 1, 1))) == (self.s, null)

    def testWithOptionalAndDefaultedIndefMode(self):
        assert decoder.decode(ints2octs((49, 128, 5, 0, 36, 128, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 0, 0, 2, 1, 1, 0, 0))) == (self.s, null)

    def testWithOptionalAndDefaultedDefModeChunked(self):
        assert decoder.decode(ints2octs((49, 24, 5, 0, 36, 17, 4, 4, 113, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 3, 111, 119, 110, 2, 1, 1))) == (self.s, null)

    def testWithOptionalAndDefaultedIndefModeChunked(self):
        assert decoder.decode(ints2octs((49, 128, 5, 0, 36, 128, 4, 4, 113, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 3, 111, 119, 110, 0, 0, 2, 1, 1, 0, 0))) == (self.s, null)

    def testWithOptionalAndDefaultedDefModeSubst(self):
        assert decoder.decode(ints2octs((49, 18, 5, 0, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 2, 1, 1)), substrateFun=lambda a, b, c: (b, b[c:])) == (ints2octs((5, 0, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 2, 1, 1)), str2octs(''))

    def testWithOptionalAndDefaultedIndefModeSubst(self):
        assert decoder.decode(ints2octs((49, 128, 5, 0, 36, 128, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 0, 0, 2, 1, 1, 0, 0)), substrateFun=lambda a, b, c: (b, str2octs(''))) == (ints2octs((5, 0, 36, 128, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 0, 0, 2, 1, 1, 0, 0)), str2octs(''))

    def testTagFormat(self):
        try:
            decoder.decode(ints2octs((16, 18, 5, 0, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 2, 1, 1)))
        except PyAsn1Error:
            pass
        else:
            assert 0, 'wrong tagFormat worked out'