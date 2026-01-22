from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class CountedArrayTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Word, nums, OneOrMore, countedArray
        testString = '2 5 7 6 0 1 2 3 4 5 0 3 5 4 3'
        integer = Word(nums).setParseAction(lambda t: int(t[0]))
        countedField = countedArray(integer)
        r = OneOrMore(countedField).parseString(testString)
        print_(testString)
        print_(r.asList())
        self.assertEqual(r.asList(), [[5, 7], [0, 1, 2, 3, 4, 5], [], [5, 4, 3]], 'Failed matching countedArray, got ' + str(r.asList()))