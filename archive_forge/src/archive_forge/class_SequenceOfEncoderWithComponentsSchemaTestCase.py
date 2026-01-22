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
class SequenceOfEncoderWithComponentsSchemaTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.SequenceOf(componentType=univ.OctetString())

    def __init(self):
        self.s.clear()
        self.s.setComponentByPosition(0, 'quick brown')

    def testDefMode(self):
        self.__init()
        assert encoder.encode(self.s) == ints2octs((48, 13, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110))

    def testIndefMode(self):
        self.__init()
        assert encoder.encode(self.s, defMode=False) == ints2octs((48, 128, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 0, 0))

    def testDefModeChunked(self):
        self.__init()
        assert encoder.encode(self.s, defMode=True, maxChunkSize=4) == ints2octs((48, 19, 36, 17, 4, 4, 113, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 3, 111, 119, 110))

    def testIndefModeChunked(self):
        self.__init()
        assert encoder.encode(self.s, defMode=False, maxChunkSize=4) == ints2octs((48, 128, 36, 128, 4, 4, 113, 117, 105, 99, 4, 4, 107, 32, 98, 114, 4, 3, 111, 119, 110, 0, 0, 0, 0))