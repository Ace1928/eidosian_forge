import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.native import encoder
from pyasn1.compat.octets import str2octs
from pyasn1.error import PyAsn1Error
class ChoiceEncoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.s = univ.Choice(componentType=namedtype.NamedTypes(namedtype.NamedType('place-holder', univ.Null('')), namedtype.NamedType('number', univ.Integer(0)), namedtype.NamedType('string', univ.OctetString())))

    def testEmpty(self):
        try:
            encoder.encode(self.s)
        except PyAsn1Error:
            pass
        else:
            assert False, 'encoded unset choice'

    def testFilled(self):
        self.s.setComponentByPosition(0, univ.Null(''))
        assert encoder.encode(self.s) == {'place-holder': None}