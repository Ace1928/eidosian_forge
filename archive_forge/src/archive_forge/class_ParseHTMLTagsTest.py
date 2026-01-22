from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseHTMLTagsTest(ParseTestCase):

    def runTest(self):
        test = '\n            <BODY>\n            <BODY BGCOLOR="#00FFCC">\n            <BODY BGCOLOR="#00FFAA"/>\n            <BODY BGCOLOR=\'#00FFBB\' FGCOLOR=black>\n            <BODY/>\n            </BODY>\n        '
        results = [('startBody', False, '', ''), ('startBody', False, '#00FFCC', ''), ('startBody', True, '#00FFAA', ''), ('startBody', False, '#00FFBB', 'black'), ('startBody', True, '', ''), ('endBody', False, '', '')]
        bodyStart, bodyEnd = pp.makeHTMLTags('BODY')
        resIter = iter(results)
        for t, s, e in (bodyStart | bodyEnd).scanString(test):
            print_(test[s:e], '->', t.asList())
            expectedType, expectedEmpty, expectedBG, expectedFG = next(resIter)
            print_(t.dump())
            if 'startBody' in t:
                self.assertEqual(bool(t.empty), expectedEmpty, 'expected %s token, got %s' % (expectedEmpty and 'empty' or 'not empty', t.empty and 'empty' or 'not empty'))
                self.assertEqual(t.bgcolor, expectedBG, 'failed to match BGCOLOR, expected %s, got %s' % (expectedBG, t.bgcolor))
                self.assertEqual(t.fgcolor, expectedFG, 'failed to match FGCOLOR, expected %s, got %s' % (expectedFG, t.bgcolor))
            elif 'endBody' in t:
                print_('end tag')
                pass
            else:
                print_('BAD!!!')