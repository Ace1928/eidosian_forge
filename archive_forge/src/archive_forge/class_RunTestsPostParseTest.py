from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class RunTestsPostParseTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        integer = pp.pyparsing_common.integer
        fraction = integer('numerator') + '/' + integer('denominator')
        accum = []

        def eval_fraction(test, result):
            accum.append((test, result.asList()))
            return 'eval: {0}'.format(result.numerator / result.denominator)
        success = fraction.runTests('            1/2\n            1/0\n        ', postParse=eval_fraction)[0]
        print_(success)
        self.assertTrue(success, 'failed to parse fractions in RunTestsPostParse')
        expected_accum = [('1/2', [1, '/', 2]), ('1/0', [1, '/', 0])]
        self.assertEqual(accum, expected_accum, 'failed to call postParse method during runTests')