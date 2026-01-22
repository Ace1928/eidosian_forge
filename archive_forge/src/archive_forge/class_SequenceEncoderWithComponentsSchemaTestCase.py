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
class SequenceEncoderWithComponentsSchemaTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.NamedType('place-holder', univ.Null()), namedtype.OptionalNamedType('first-name', univ.OctetString()), namedtype.DefaultedNamedType('age', univ.Integer(33))))

    def __init(self):
        self.s.clear()
        self.s.setComponentByPosition(0, '')

    def __initWithOptional(self):
        self.s.clear()
        self.s.setComponentByPosition(0, '')
        self.s.setComponentByPosition(1, 'quick brown')

    def __initWithDefaulted(self):
        self.s.clear()
        self.s.setComponentByPosition(0, '')
        self.s.setComponentByPosition(2, 1)

    def __initWithOptionalAndDefaulted(self):
        self.s.clear()
        self.s.setComponentByPosition(0, univ.Null(''))
        self.s.setComponentByPosition(1, univ.OctetString('quick brown'))
        self.s.setComponentByPosition(2, univ.Integer(1))

    def testDefMode(self):
        self.__init()
        assert encoder.encode(self.s) == ints2octs((48, 2, 5, 0))

    def testIndefMode(self):
        self.__init()
        assert encoder.encode(self.s, defMode=False) == ints2octs((48, 128, 5, 0, 0, 0))

    def testDefModeChunked(self):
        self.__init()
        assert encoder.encode(self.s, defMode=True, maxChunkSize=4) == ints2octs((48, 2, 5, 0))

    def testIndefModeChunked(self):
        self.__init()
        assert encoder.encode(self.s, defMode=False, maxChunkSize=4) == ints2octs((48, 128, 5, 0, 0, 0))

    def testWithOptionalDefMode(self):
        self.__initWithOptional()
        assert encoder.encode(self.s) == ints2octs((48, 15, 5, 0, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110))

    def testWithOptionalIndefMode(self):
        self.__initWithOptional()
        assert encoder.encode(self.s, defMode=False) == ints2octs((48, 128, 5, 0, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 0, 0))

    def testWithOptionalDefModeChunked(self):
        self.__initWithOptional()
        assert encoder.encode(self.s, defMode=True, maxChunkSize=4) == ints2octs((48, 21, 5, 0, 36, 17, 4, 4, 113, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 3, 111, 119, 110))

    def testWithOptionalIndefModeChunked(self):
        self.__initWithOptional()
        assert encoder.encode(self.s, defMode=False, maxChunkSize=4) == ints2octs((48, 128, 5, 0, 36, 128, 4, 4, 113, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 3, 111, 119, 110, 0, 0, 0, 0))

    def testWithDefaultedDefMode(self):
        self.__initWithDefaulted()
        assert encoder.encode(self.s) == ints2octs((48, 5, 5, 0, 2, 1, 1))

    def testWithDefaultedIndefMode(self):
        self.__initWithDefaulted()
        assert encoder.encode(self.s, defMode=False) == ints2octs((48, 128, 5, 0, 2, 1, 1, 0, 0))

    def testWithDefaultedDefModeChunked(self):
        self.__initWithDefaulted()
        assert encoder.encode(self.s, defMode=True, maxChunkSize=4) == ints2octs((48, 5, 5, 0, 2, 1, 1))

    def testWithDefaultedIndefModeChunked(self):
        self.__initWithDefaulted()
        assert encoder.encode(self.s, defMode=False, maxChunkSize=4) == ints2octs((48, 128, 5, 0, 2, 1, 1, 0, 0))

    def testWithOptionalAndDefaultedDefMode(self):
        self.__initWithOptionalAndDefaulted()
        assert encoder.encode(self.s) == ints2octs((48, 18, 5, 0, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 2, 1, 1))

    def testWithOptionalAndDefaultedIndefMode(self):
        self.__initWithOptionalAndDefaulted()
        assert encoder.encode(self.s, defMode=False) == ints2octs((48, 128, 5, 0, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 2, 1, 1, 0, 0))

    def testWithOptionalAndDefaultedDefModeChunked(self):
        self.__initWithOptionalAndDefaulted()
        assert encoder.encode(self.s, defMode=True, maxChunkSize=4) == ints2octs((48, 24, 5, 0, 36, 17, 4, 4, 113, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 3, 111, 119, 110, 2, 1, 1))

    def testWithOptionalAndDefaultedIndefModeChunked(self):
        self.__initWithOptionalAndDefaulted()
        assert encoder.encode(self.s, defMode=False, maxChunkSize=4) == ints2octs((48, 128, 5, 0, 36, 128, 4, 4, 113, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 3, 111, 119, 110, 0, 0, 2, 1, 1, 0, 0))