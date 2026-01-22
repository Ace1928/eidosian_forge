from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseResultsWithNameMatchFirst(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        expr_a = pp.Literal('not') + pp.Literal('the') + pp.Literal('bird')
        expr_b = pp.Literal('the') + pp.Literal('bird')
        expr = (expr_a | expr_b)('rexp')
        expr.runTests('            not the bird\n            the bird\n        ')
        self.assertEqual(list(expr.parseString('not the bird')['rexp']), 'not the bird'.split())
        self.assertEqual(list(expr.parseString('the bird')['rexp']), 'the bird'.split())
        with AutoReset(pp.__compat__, 'collect_all_And_tokens'):
            pp.__compat__.collect_all_And_tokens = False
            pp.__diag__.warn_multiple_tokens_in_named_alternation = True
            expr_a = pp.Literal('not') + pp.Literal('the') + pp.Literal('bird')
            expr_b = pp.Literal('the') + pp.Literal('bird')
            if PY_3:
                with self.assertWarns(UserWarning, msg='failed to warn of And within alternation'):
                    expr = (expr_a | expr_b)('rexp')
            else:
                self.expect_warning = True
                expr = (expr_a | expr_b)('rexp')
            expr.runTests('\n                not the bird\n                the bird\n            ')
            self.assertEqual(expr.parseString('not the bird')['rexp'], 'not')
            self.assertEqual(expr.parseString('the bird')['rexp'], 'the')