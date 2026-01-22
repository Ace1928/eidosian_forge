from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class UnicodeExpressionTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Literal, ParseException
        z = 'a' | Literal(u'ᄑ')
        z.streamline()
        try:
            z.parseString('b')
        except ParseException as pe:
            if not PY_3:
                self.assertEqual(pe.msg, 'Expected {"a" | "\\u1111"}', 'Invalid error message raised, got %r' % pe.msg)
            else:
                self.assertEqual(pe.msg, 'Expected {"a" | "ᄑ"}', 'Invalid error message raised, got %r' % pe.msg)