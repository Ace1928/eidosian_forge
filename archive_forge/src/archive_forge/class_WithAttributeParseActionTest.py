from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class WithAttributeParseActionTest(ParseTestCase):

    def runTest(self):
        """
        This unit test checks withAttribute in these ways:

        * Argument forms as keywords and tuples
        * Selecting matching tags by attribute
        * Case-insensitive attribute matching
        * Correctly matching tags having the attribute, and rejecting tags not having the attribute

        (Unit test written by voigts as part of the Google Highly Open Participation Contest)
        """
        from pyparsing import makeHTMLTags, Word, withAttribute, withClass, nums
        data = '\n        <a>1</a>\n        <a b="x">2</a>\n        <a B="x">3</a>\n        <a b="X">4</a>\n        <a b="y">5</a>\n        <a class="boo">8</ a>\n        '
        tagStart, tagEnd = makeHTMLTags('a')
        expr = tagStart + Word(nums)('value') + tagEnd
        expected = ([['a', ['b', 'x'], False, '2', '</a>'], ['a', ['b', 'x'], False, '3', '</a>']], [['a', ['b', 'x'], False, '2', '</a>'], ['a', ['b', 'x'], False, '3', '</a>']], [['a', ['class', 'boo'], False, '8', '</a>']])
        for attrib, exp in zip([withAttribute(b='x'), withAttribute(('b', 'x')), withClass('boo')], expected):
            tagStart.setParseAction(attrib)
            result = expr.searchString(data)
            print_(result.dump())
            self.assertEqual(result.asList(), exp, 'Failed test, expected %s, got %s' % (expected, result.asList()))