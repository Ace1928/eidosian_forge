from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseActionNestingTest(ParseTestCase):

    def runTest(self):
        vals = pp.OneOrMore(pp.pyparsing_common.integer)('int_values')

        def add_total(tokens):
            tokens['total'] = sum(tokens)
            return tokens
        vals.addParseAction(add_total)
        results = vals.parseString('244 23 13 2343')
        print_(results.dump())
        self.assertEqual(results.int_values.asDict(), {}, 'noop parse action changed ParseResults structure')
        name = pp.Word(pp.alphas)('name')
        score = pp.Word(pp.nums + '.')('score')
        nameScore = pp.Group(name + score)
        line1 = nameScore('Rider')
        result1 = line1.parseString('Mauney 46.5')
        print_('### before parse action is added ###')
        print_('result1.dump():\n' + result1.dump() + '\n')
        before_pa_dict = result1.asDict()
        line1.setParseAction(lambda t: t)
        result1 = line1.parseString('Mauney 46.5')
        after_pa_dict = result1.asDict()
        print_('### after parse action was added ###')
        print_('result1.dump():\n' + result1.dump() + '\n')
        self.assertEqual(before_pa_dict, after_pa_dict, 'noop parse action changed ParseResults structure')