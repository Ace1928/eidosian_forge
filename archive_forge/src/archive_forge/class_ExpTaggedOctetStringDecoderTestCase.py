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
class ExpTaggedOctetStringDecoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.o = univ.OctetString('Quick brown fox', tagSet=univ.OctetString.tagSet.tagExplicitly(tag.Tag(tag.tagClassApplication, tag.tagFormatSimple, 5)))

    def testDefMode(self):
        o, r = decoder.decode(ints2octs((101, 17, 4, 15, 81, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120)))
        assert not r
        assert self.o == o
        assert self.o.tagSet == o.tagSet
        assert self.o.isSameTypeWith(o)

    def testIndefMode(self):
        o, r = decoder.decode(ints2octs((101, 128, 36, 128, 4, 15, 81, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120, 0, 0, 0, 0)))
        assert not r
        assert self.o == o
        assert self.o.tagSet == o.tagSet
        assert self.o.isSameTypeWith(o)

    def testDefModeChunked(self):
        o, r = decoder.decode(ints2octs((101, 25, 36, 23, 4, 4, 81, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 4, 111, 119, 110, 32, 4, 3, 102, 111, 120)))
        assert not r
        assert self.o == o
        assert self.o.tagSet == o.tagSet
        assert self.o.isSameTypeWith(o)

    def testIndefModeChunked(self):
        o, r = decoder.decode(ints2octs((101, 128, 36, 128, 4, 4, 81, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 4, 111, 119, 110, 32, 4, 3, 102, 111, 120, 0, 0, 0, 0)))
        assert not r
        assert self.o == o
        assert self.o.tagSet == o.tagSet
        assert self.o.isSameTypeWith(o)

    def testDefModeSubst(self):
        assert decoder.decode(ints2octs((101, 17, 4, 15, 81, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120)), substrateFun=lambda a, b, c: (b, b[c:])) == (ints2octs((4, 15, 81, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120)), str2octs(''))

    def testIndefModeSubst(self):
        assert decoder.decode(ints2octs((101, 128, 36, 128, 4, 15, 81, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120, 0, 0, 0, 0)), substrateFun=lambda a, b, c: (b, str2octs(''))) == (ints2octs((36, 128, 4, 15, 81, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120, 0, 0, 0, 0)), str2octs(''))