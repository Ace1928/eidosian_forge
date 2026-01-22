import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class NestedOptionalSequenceOfEncoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        layer2 = univ.SequenceOf(componentType=univ.OctetString())
        layer1 = univ.Sequence(componentType=namedtype.NamedTypes(namedtype.OptionalNamedType('inner', layer2)))
        self.s = layer1

    def __initOptionalWithValue(self):
        self.s.clear()
        self.s[0][0] = 'test'
        return self.s

    def __initOptional(self):
        self.s.clear()
        return self.s

    def testOptionalWithValue(self):
        s = self.__initOptionalWithValue()
        assert encoder.encode(s) == ints2octs((48, 128, 48, 128, 4, 4, 116, 101, 115, 116, 0, 0, 0, 0))

    def testOptional(self):
        s = self.__initOptional()
        assert encoder.encode(s) == ints2octs((48, 128, 0, 0))