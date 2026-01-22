from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class SetNameTest(ParseTestCase):

    def runTest(self):
        from pyparsing import oneOf, infixNotation, Word, nums, opAssoc, delimitedList, countedArray, nestedExpr, makeHTMLTags, anyOpenTag, anyCloseTag, commonHTMLEntity, replaceHTMLEntity, Forward, ZeroOrMore
        a = oneOf('a b c')
        b = oneOf('d e f')
        arith_expr = infixNotation(Word(nums), [(oneOf('* /'), 2, opAssoc.LEFT), (oneOf('+ -'), 2, opAssoc.LEFT)])
        arith_expr2 = infixNotation(Word(nums), [(('?', ':'), 3, opAssoc.LEFT)])
        recursive = Forward()
        recursive <<= a + ZeroOrMore(b + recursive)
        tests = [a, b, a | b, arith_expr, arith_expr.expr, arith_expr2, arith_expr2.expr, recursive, delimitedList(Word(nums).setName('int')), countedArray(Word(nums).setName('int')), nestedExpr(), makeHTMLTags('Z'), (anyOpenTag, anyCloseTag), commonHTMLEntity, commonHTMLEntity.setParseAction(replaceHTMLEntity).transformString('lsdjkf &lt;lsdjkf&gt;&amp;&apos;&quot;&xyzzy;')]
        expected = map(str.strip, '            a | b | c\n            d | e | f\n            {a | b | c | d | e | f}\n            Forward: + | - term\n            + | - term\n            Forward: ?: term\n            ?: term\n            Forward: {a | b | c [{d | e | f : ...}]...}\n            int [, int]...\n            (len) int...\n            nested () expression\n            (<Z>, </Z>)\n            (<any tag>, </any tag>)\n            common HTML entity\n            lsdjkf <lsdjkf>&\'"&xyzzy;'.splitlines())
        for t, e in zip(tests, expected):
            tname = str(t)
            print_(tname)
            self.assertEqual(tname, e, 'expression name mismatch, expected {0} got {1}'.format(e, tname))