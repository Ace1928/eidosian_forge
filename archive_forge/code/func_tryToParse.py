from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
def tryToParse(someText, fail_expected=False):
    try:
        print_(testExpr.parseString(someText))
        self.assertFalse(fail_expected, 'expected failure but no exception raised')
    except Exception as e:
        print_('Exception %s while parsing string %s' % (e, repr(someText)))
        self.assertTrue(fail_expected and isinstance(e, ParseBaseException), 'Exception %s while parsing string %s' % (e, repr(someText)))