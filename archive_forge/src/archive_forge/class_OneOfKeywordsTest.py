from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class OneOfKeywordsTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        literal_expr = pp.oneOf('a b c')
        success, _ = literal_expr[...].runTests('\n            # literal oneOf tests\n            a b c\n            a a a\n            abc\n        ')
        self.assertTrue(success, 'failed literal oneOf matching')
        keyword_expr = pp.oneOf('a b c', asKeyword=True)
        success, _ = keyword_expr[...].runTests('\n            # keyword oneOf tests\n            a b c\n            a a a\n        ')
        self.assertTrue(success, 'failed keyword oneOf matching')
        success, _ = keyword_expr[...].runTests('\n            # keyword oneOf failure tests\n            abc\n        ', failureTests=True)
        self.assertTrue(success, 'failed keyword oneOf failure tests')