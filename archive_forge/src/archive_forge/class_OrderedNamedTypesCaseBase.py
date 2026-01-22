import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.error import PyAsn1Error
class OrderedNamedTypesCaseBase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.e = namedtype.NamedTypes(namedtype.NamedType('first-name', univ.OctetString('')), namedtype.NamedType('age', univ.Integer(0)))

    def testGetTypeByPosition(self):
        assert self.e.getTypeByPosition(0) == univ.OctetString(''), 'getTypeByPosition() fails'