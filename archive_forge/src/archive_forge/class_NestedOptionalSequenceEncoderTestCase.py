import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class NestedOptionalSequenceEncoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        inner = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.OptionalNamedType('first-name', univ.OctetString()), namedtype.DefaultedNamedType('age', univ.Integer(33))))
        outerWithOptional = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.OptionalNamedType('inner', inner)))
        outerWithDefault = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.DefaultedNamedType('inner', inner)))
        self.s1 = outerWithOptional
        self.s2 = outerWithDefault

    def __initOptionalWithDefaultAndOptional(self):
        self.s1.clear()
        self.s1[0][0] = 'test'
        self.s1[0][1] = 123
        return self.s1

    def __initOptionalWithDefault(self):
        self.s1.clear()
        self.s1[0][1] = 123
        return self.s1

    def __initOptionalWithOptional(self):
        self.s1.clear()
        self.s1[0][0] = 'test'
        return self.s1

    def __initOptional(self):
        self.s1.clear()
        return self.s1

    def __initDefaultWithDefaultAndOptional(self):
        self.s2.clear()
        self.s2[0][0] = 'test'
        self.s2[0][1] = 123
        return self.s2

    def __initDefaultWithDefault(self):
        self.s2.clear()
        self.s2[0][0] = 'test'
        return self.s2

    def __initDefaultWithOptional(self):
        self.s2.clear()
        self.s2[0][1] = 123
        return self.s2

    def testOptionalWithDefaultAndOptional(self):
        s = self.__initOptionalWithDefaultAndOptional()
        assert encoder.encode(s) == ints2octs((48, 128, 48, 128, 4, 4, 116, 101, 115, 116, 2, 1, 123, 0, 0, 0, 0))

    def testOptionalWithDefault(self):
        s = self.__initOptionalWithDefault()
        assert encoder.encode(s) == ints2octs((48, 128, 48, 128, 2, 1, 123, 0, 0, 0, 0))

    def testOptionalWithOptional(self):
        s = self.__initOptionalWithOptional()
        assert encoder.encode(s) == ints2octs((48, 128, 48, 128, 4, 4, 116, 101, 115, 116, 0, 0, 0, 0))

    def testOptional(self):
        s = self.__initOptional()
        assert encoder.encode(s) == ints2octs((48, 128, 0, 0))

    def testDefaultWithDefaultAndOptional(self):
        s = self.__initDefaultWithDefaultAndOptional()
        assert encoder.encode(s) == ints2octs((48, 128, 48, 128, 4, 4, 116, 101, 115, 116, 2, 1, 123, 0, 0, 0, 0))

    def testDefaultWithDefault(self):
        s = self.__initDefaultWithDefault()
        assert encoder.encode(s) == ints2octs((48, 128, 48, 128, 4, 4, 116, 101, 115, 116, 0, 0, 0, 0))

    def testDefaultWithOptional(self):
        s = self.__initDefaultWithOptional()
        assert encoder.encode(s) == ints2octs((48, 128, 48, 128, 2, 1, 123, 0, 0, 0, 0))