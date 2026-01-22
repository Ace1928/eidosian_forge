from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class RecursiveCombineTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Forward, Word, alphas, nums, Optional, Combine
        testInput = 'myc(114)r(11)dd'
        Stream = Forward()
        Stream << Optional(Word(alphas)) + Optional('(' + Word(nums) + ')' + Stream)
        expected = Stream.parseString(testInput).asList()
        print_([''.join(expected)])
        Stream = Forward()
        Stream << Combine(Optional(Word(alphas)) + Optional('(' + Word(nums) + ')' + Stream))
        testVal = Stream.parseString(testInput).asList()
        print_(testVal)
        self.assertEqual(''.join(testVal), ''.join(expected), 'Failed to process Combine with recursive content')