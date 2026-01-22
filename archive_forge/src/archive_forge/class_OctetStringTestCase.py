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
class OctetStringTestCase(BaseTestCase):

    def testBinDefault(self):

        class BinDefault(univ.OctetString):
            defaultBinValue = '1000010111101110101111000000111011'
        assert BinDefault() == univ.OctetString(binValue='1000010111101110101111000000111011')

    def testHexDefault(self):

        class HexDefault(univ.OctetString):
            defaultHexValue = 'FA9823C43E43510DE3422'
        assert HexDefault() == univ.OctetString(hexValue='FA9823C43E43510DE3422')

    def testBinStr(self):
        assert univ.OctetString(binValue='1000010111101110101111000000111011') == ints2octs((133, 238, 188, 14, 192)), 'bin init fails'

    def testHexStr(self):
        assert univ.OctetString(hexValue='FA9823C43E43510DE3422') == ints2octs((250, 152, 35, 196, 62, 67, 81, 13, 227, 66, 32)), 'hex init fails'

    def testTuple(self):
        assert univ.OctetString((1, 2, 3, 4, 5)) == ints2octs((1, 2, 3, 4, 5)), 'tuple init failed'

    def testRepr(self):
        assert 'abc' in repr(univ.OctetString('abc'))

    def testEmpty(self):
        try:
            str(univ.OctetString())
        except PyAsn1Error:
            pass
        else:
            assert 0, 'empty OctetString() not reported'

    def testTag(self):
        assert univ.OctetString().tagSet == tag.TagSet((), tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 4))

    def testStaticDef(self):

        class OctetString(univ.OctetString):
            pass
        assert OctetString(hexValue='FA9823C43E43510DE3422') == ints2octs((250, 152, 35, 196, 62, 67, 81, 13, 227, 66, 32))