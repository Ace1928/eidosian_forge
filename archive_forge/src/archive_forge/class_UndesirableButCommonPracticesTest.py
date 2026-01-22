from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class UndesirableButCommonPracticesTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        ppc = pp.pyparsing_common
        expr = pp.And([pp.Word('abc'), pp.Word('123')])
        expr.runTests('\n            aaa 333\n            b 1\n            ababab 32123\n        ')
        expr = pp.Or(pp.Or(ppc.integer))
        expr.runTests('\n            123\n            456\n            abc\n        ')