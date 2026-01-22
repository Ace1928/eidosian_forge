import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
class SetWithChoiceWithSchemaEncoderTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        c = univ.Choice(componentType=namedtype.NamedTypes(namedtype.NamedType('actual', univ.Boolean(0))))
        self.s = univ.Set(componentType=namedtype.NamedTypes(namedtype.NamedType('place-holder', univ.Null('')), namedtype.NamedType('status', c)))

    def testIndefMode(self):
        self.s.setComponentByPosition(0)
        self.s.setComponentByName('status')
        self.s.getComponentByName('status').setComponentByPosition(0, 1)
        assert encoder.encode(self.s) == ints2octs((49, 128, 1, 1, 255, 5, 0, 0, 0))