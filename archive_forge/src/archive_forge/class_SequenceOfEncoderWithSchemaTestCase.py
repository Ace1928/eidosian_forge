import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class SequenceOfEncoderWithSchemaTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.SequenceOf(componentType=univ.OctetString())

    def testEmpty(self):
        self.s.clear()
        assert encoder.encode(self.s) == ints2octs((48, 128, 0, 0))

    def testIndefMode1(self):
        self.s.clear()
        self.s.append('a')
        self.s.append('ab')
        assert encoder.encode(self.s) == ints2octs((48, 128, 4, 1, 97, 4, 2, 97, 98, 0, 0))

    def testIndefMode2(self):
        self.s.clear()
        self.s.append('ab')
        self.s.append('a')
        assert encoder.encode(self.s) == ints2octs((48, 128, 4, 2, 97, 98, 4, 1, 97, 0, 0))

    def testIndefMode3(self):
        self.s.clear()
        self.s.append('b')
        self.s.append('a')
        assert encoder.encode(self.s) == ints2octs((48, 128, 4, 1, 98, 4, 1, 97, 0, 0))

    def testIndefMode4(self):
        self.s.clear()
        self.s.append('a')
        self.s.append('b')
        assert encoder.encode(self.s) == ints2octs((48, 128, 4, 1, 97, 4, 1, 98, 0, 0))