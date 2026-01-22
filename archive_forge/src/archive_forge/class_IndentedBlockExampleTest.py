from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class IndentedBlockExampleTest(ParseTestCase):

    def runTest(self):
        from textwrap import dedent
        from pyparsing import Word, alphas, alphanums, indentedBlock, Optional, delimitedList, Group, Forward, nums, OneOrMore
        data = dedent('\n        def A(z):\n          A1\n          B = 100\n          G = A2\n          A2\n          A3\n        B\n        def BB(a,b,c):\n          BB1\n          def BBA():\n            bba1\n            bba2\n            bba3\n        C\n        D\n        def spam(x,y):\n             def eggs(z):\n                 pass\n        ')
        indentStack = [1]
        stmt = Forward()
        identifier = Word(alphas, alphanums)
        funcDecl = 'def' + identifier + Group('(' + Optional(delimitedList(identifier)) + ')') + ':'
        func_body = indentedBlock(stmt, indentStack)
        funcDef = Group(funcDecl + func_body)
        rvalue = Forward()
        funcCall = Group(identifier + '(' + Optional(delimitedList(rvalue)) + ')')
        rvalue << (funcCall | identifier | Word(nums))
        assignment = Group(identifier + '=' + rvalue)
        stmt << (funcDef | assignment | identifier)
        module_body = OneOrMore(stmt)
        parseTree = module_body.parseString(data)
        parseTree.pprint()
        self.assertEqual(parseTree.asList(), [['def', 'A', ['(', 'z', ')'], ':', [['A1'], [['B', '=', '100']], [['G', '=', 'A2']], ['A2'], ['A3']]], 'B', ['def', 'BB', ['(', 'a', 'b', 'c', ')'], ':', [['BB1'], [['def', 'BBA', ['(', ')'], ':', [['bba1'], ['bba2'], ['bba3']]]]]], 'C', 'D', ['def', 'spam', ['(', 'x', 'y', ')'], ':', [[['def', 'eggs', ['(', 'z', ')'], ':', [['pass']]]]]]], 'Failed indentedBlock example')