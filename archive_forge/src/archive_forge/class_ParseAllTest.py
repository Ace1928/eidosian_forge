from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseAllTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Word, cppStyleComment
        testExpr = Word('A')
        tests = [('AAAAA', False, True), ('AAAAA', True, True), ('AAABB', False, True), ('AAABB', True, False)]
        for s, parseAllFlag, shouldSucceed in tests:
            try:
                print_("'%s' parseAll=%s (shouldSuceed=%s)" % (s, parseAllFlag, shouldSucceed))
                testExpr.parseString(s, parseAllFlag)
                self.assertTrue(shouldSucceed, 'successfully parsed when should have failed')
            except ParseException as pe:
                self.assertFalse(shouldSucceed, 'failed to parse when should have succeeded')
        testExpr.ignore(cppStyleComment)
        tests = [('AAAAA //blah', False, True), ('AAAAA //blah', True, True), ('AAABB //blah', False, True), ('AAABB //blah', True, False)]
        for s, parseAllFlag, shouldSucceed in tests:
            try:
                print_("'%s' parseAll=%s (shouldSucceed=%s)" % (s, parseAllFlag, shouldSucceed))
                testExpr.parseString(s, parseAllFlag)
                self.assertTrue(shouldSucceed, 'successfully parsed when should have failed')
            except ParseException as pe:
                self.assertFalse(shouldSucceed, 'failed to parse when should have succeeded')