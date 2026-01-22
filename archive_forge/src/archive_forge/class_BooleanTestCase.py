import math
import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import univ
from pyasn1.type import tag
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import error
from pyasn1.compat.octets import str2octs, ints2octs, octs2ints
from pyasn1.error import PyAsn1Error
class BooleanTestCase(BaseTestCase):

    def testTruth(self):
        assert univ.Boolean(True) and univ.Boolean(1), 'Truth initializer fails'

    def testFalse(self):
        assert not univ.Boolean(False) and (not univ.Boolean(0)), 'False initializer fails'

    def testStr(self):
        assert str(univ.Boolean(1)) == 'True', 'str() fails'

    def testInt(self):
        assert int(univ.Boolean(1)) == 1, 'int() fails'

    def testRepr(self):
        assert 'Boolean' in repr(univ.Boolean(1))

    def testTag(self):
        assert univ.Boolean().tagSet == tag.TagSet((), tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 1))

    def testConstraints(self):

        class Boolean(univ.Boolean):
            pass
        try:
            Boolean(2)
        except error.ValueConstraintError:
            pass
        else:
            assert 0, 'constraint fail'