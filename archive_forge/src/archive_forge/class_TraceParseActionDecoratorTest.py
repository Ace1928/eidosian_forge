from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class TraceParseActionDecoratorTest(ParseTestCase):

    def runTest(self):
        from pyparsing import traceParseAction, Word, nums

        @traceParseAction
        def convert_to_int(t):
            return int(t[0])

        class Z(object):

            def __call__(self, other):
                return other[0] * 1000
        integer = Word(nums).addParseAction(convert_to_int)
        integer.addParseAction(traceParseAction(lambda t: t[0] * 10))
        integer.addParseAction(traceParseAction(Z()))
        integer.parseString('132')