from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseExpressionResultsTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Word, alphas, OneOrMore, Optional, Group
        a = Word('a', alphas).setName('A')
        b = Word('b', alphas).setName('B')
        c = Word('c', alphas).setName('C')
        ab = (a + b).setName('AB')
        abc = (ab + c).setName('ABC')
        word = Word(alphas).setName('word')
        words = Group(OneOrMore(~a + word)).setName('words')
        phrase = words('Head') + Group(a + Optional(b + Optional(c)))('ABC') + words('Tail')
        results = phrase.parseString('xavier yeti alpha beta charlie will beaver')
        print_(results, results.Head, results.ABC, results.Tail)
        for key, ln in [('Head', 2), ('ABC', 3), ('Tail', 2)]:
            self.assertEqual(len(results[key]), ln, 'expected %d elements in %s, found %s' % (ln, key, str(results[key])))