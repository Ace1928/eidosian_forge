from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class TokenMapTest(ParseTestCase):

    def runTest(self):
        from pyparsing import tokenMap, Word, hexnums, OneOrMore
        parser = OneOrMore(Word(hexnums)).setParseAction(tokenMap(int, 16))
        success, results = parser.runTests('\n            00 11 22 aa FF 0a 0d 1a\n            ', printResults=False)
        self.assertTrue(success, 'failed to parse hex integers')
        print_(results)
        self.assertEqual(results[0][-1].asList(), [0, 17, 34, 170, 255, 10, 13, 26], 'tokenMap parse action failed')