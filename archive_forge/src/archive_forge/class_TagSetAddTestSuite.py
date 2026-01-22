import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
class TagSetAddTestSuite(TagSetTestCaseBase):

    def testAdd(self):
        t = self.ts1 + tag.Tag(tag.tagClassApplication, tag.tagFormatSimple, 2)
        assert t == tag.TagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12), tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12), tag.Tag(tag.tagClassApplication, tag.tagFormatSimple, 2)), 'TagSet.__add__() fails'

    def testRadd(self):
        t = tag.Tag(tag.tagClassApplication, tag.tagFormatSimple, 2) + self.ts1
        assert t == tag.TagSet(tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12), tag.Tag(tag.tagClassApplication, tag.tagFormatSimple, 2), tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12)), 'TagSet.__radd__() fails'