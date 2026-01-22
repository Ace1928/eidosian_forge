from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class InfixNotationGrammarTest2(ParseTestCase):

    def runTest(self):
        from pyparsing import infixNotation, Word, alphas, oneOf, opAssoc
        boolVars = {'True': True, 'False': False}

        class BoolOperand(object):
            reprsymbol = ''

            def __init__(self, t):
                self.args = t[0][0::2]

            def __str__(self):
                sep = ' %s ' % self.reprsymbol
                return '(' + sep.join(map(str, self.args)) + ')'

        class BoolAnd(BoolOperand):
            reprsymbol = '&'

            def __bool__(self):
                for a in self.args:
                    if isinstance(a, str):
                        v = boolVars[a]
                    else:
                        v = bool(a)
                    if not v:
                        return False
                return True

        class BoolOr(BoolOperand):
            reprsymbol = '|'

            def __bool__(self):
                for a in self.args:
                    if isinstance(a, str):
                        v = boolVars[a]
                    else:
                        v = bool(a)
                    if v:
                        return True
                return False

        class BoolNot(BoolOperand):

            def __init__(self, t):
                self.arg = t[0][1]

            def __str__(self):
                return '~' + str(self.arg)

            def __bool__(self):
                if isinstance(self.arg, str):
                    v = boolVars[self.arg]
                else:
                    v = bool(self.arg)
                return not v
        boolOperand = Word(alphas, max=1) | oneOf('True False')
        boolExpr = infixNotation(boolOperand, [('not', 1, opAssoc.RIGHT, BoolNot), ('and', 2, opAssoc.LEFT, BoolAnd), ('or', 2, opAssoc.LEFT, BoolOr)])
        test = ['p and not q', 'not not p', 'not(p and q)', 'q or not p and r', 'q or not p or not r', 'q or not (p and r)', 'p or q or r', 'p or q or r and False', '(p or q or r) and False']
        boolVars['p'] = True
        boolVars['q'] = False
        boolVars['r'] = True
        print_('p =', boolVars['p'])
        print_('q =', boolVars['q'])
        print_('r =', boolVars['r'])
        print_()
        for t in test:
            res = boolExpr.parseString(t)[0]
            print_(t, '\n', res, '=', bool(res), '\n')