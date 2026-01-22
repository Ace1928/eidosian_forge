from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class InfixNotationGrammarTest3(ParseTestCase):

    def runTest(self):
        from pyparsing import infixNotation, Word, alphas, oneOf, opAssoc, nums, Literal
        global count
        count = 0

        def evaluate_int(t):
            global count
            value = int(t[0])
            print_('evaluate_int', value)
            count += 1
            return value
        integer = Word(nums).setParseAction(evaluate_int)
        variable = Word(alphas, exact=1)
        operand = integer | variable
        expop = Literal('^')
        signop = oneOf('+ -')
        multop = oneOf('* /')
        plusop = oneOf('+ -')
        factop = Literal('!')
        expr = infixNotation(operand, [('!', 1, opAssoc.LEFT), ('^', 2, opAssoc.LEFT), (signop, 1, opAssoc.RIGHT), (multop, 2, opAssoc.LEFT), (plusop, 2, opAssoc.LEFT)])
        test = ['9']
        for t in test:
            count = 0
            print_('%r => %s (count=%d)' % (t, expr.parseString(t), count))
            self.assertEqual(count, 1, 'count evaluated too many times!')