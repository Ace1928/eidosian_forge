import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class SetEncoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.Set()
        self.s.setComponentByPosition(0, univ.Null(''))
        self.s.setComponentByPosition(1, univ.OctetString('quick brown'))
        self.s.setComponentByPosition(2, univ.Integer(1))

    def testIndefMode(self):
        assert encoder.encode(self.s) == ints2octs((49, 128, 2, 1, 1, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 5, 0, 0, 0))

    def testWithOptionalIndefMode(self):
        assert encoder.encode(self.s) == ints2octs((49, 128, 2, 1, 1, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 5, 0, 0, 0))

    def testWithDefaultedIndefMode(self):
        assert encoder.encode(self.s) == ints2octs((49, 128, 2, 1, 1, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 5, 0, 0, 0))

    def testWithOptionalAndDefaultedIndefMode(self):
        assert encoder.encode(self.s) == ints2octs((49, 128, 2, 1, 1, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 5, 0, 0, 0))