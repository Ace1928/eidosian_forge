from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseFourFnTest(ParseTestCase):

    def runTest(self):
        import examples.fourFn as fourFn

        def test(s, ans):
            fourFn.exprStack = []
            results = fourFn.BNF().parseString(s)
            resultValue = fourFn.evaluateStack(fourFn.exprStack)
            self.assertTrue(resultValue == ans, 'failed to evaluate %s, got %f' % (s, resultValue))
            print_(s, '->', resultValue)
        from math import pi, exp
        e = exp(1)
        test('9', 9)
        test('9 + 3 + 6', 18)
        test('9 + 3 / 11', 9.0 + 3.0 / 11.0)
        test('(9 + 3)', 12)
        test('(9+3) / 11', (9.0 + 3.0) / 11.0)
        test('9 - (12 - 6)', 3)
        test('2*3.14159', 6.28318)
        test('3.1415926535*3.1415926535 / 10', 3.1415926535 * 3.1415926535 / 10.0)
        test('PI * PI / 10', pi * pi / 10.0)
        test('PI*PI/10', pi * pi / 10.0)
        test('6.02E23 * 8.048', 6.02e+23 * 8.048)
        test('e / 3', e / 3.0)
        test('sin(PI/2)', 1.0)
        test('trunc(E)', 2.0)
        test('E^PI', e ** pi)
        test('2^3^2', 2 ** 3 ** 2)
        test('2^3+2', 2 ** 3 + 2)
        test('2^9', 2 ** 9)
        test('sgn(-2)', -1)
        test('sgn(0)', 0)
        test('sgn(0.1)', 1)