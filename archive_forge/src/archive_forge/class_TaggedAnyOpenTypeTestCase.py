import sys
from tests.base import BaseTestCase
from pyasn1.type import univ
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.compat.octets import str2octs
from pyasn1.error import PyAsn1Error
class TaggedAnyOpenTypeTestCase(BaseTestCase):

    def setUp(self):
        BaseTestCase.setUp(self)
        self.taggedAny = univ.Any().subtype(implicitTag=tag.Tag(tag.tagClassPrivate, tag.tagFormatSimple, 20))

        class Sequence(univ.Sequence):
            componentType = namedtype.NamedTypes(namedtype.NamedType('id', univ.Integer()), namedtype.NamedType('blob', self.taggedAny, openType=opentype.OpenType(name='id')))
        self.s = Sequence()

    def testTypeCheckOnAssignment(self):
        self.s.clear()
        self.s['blob'] = univ.Any(str2octs('xxx'))
        self.s['blob'] = univ.Integer(123)