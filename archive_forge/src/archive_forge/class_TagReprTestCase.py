import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
class TagReprTestCase(TagTestCaseBase):

    def testRepr(self):
        assert 'Tag' in repr(self.t1)