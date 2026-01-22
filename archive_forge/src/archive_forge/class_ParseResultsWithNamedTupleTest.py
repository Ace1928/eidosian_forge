from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseResultsWithNamedTupleTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Literal, replaceWith
        expr = Literal('A')('Achar')
        expr.setParseAction(replaceWith(tuple(['A', 'Z'])))
        res = expr.parseString('A')
        print_(repr(res))
        print_(res.Achar)
        self.assertEqual(res.Achar, ('A', 'Z'), 'Failed accessing named results containing a tuple, got {0!r}'.format(res.Achar))