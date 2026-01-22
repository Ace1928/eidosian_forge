from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class PrecededByTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        num = pp.Word(pp.nums).setParseAction(lambda t: int(t[0]))
        interesting_num = pp.PrecededBy(pp.Char('abc')('prefix*')) + num
        semi_interesting_num = pp.PrecededBy('_') + num
        crazy_num = pp.PrecededBy(pp.Word('^', '$%^')('prefix*'), 10) + num
        boring_num = ~pp.PrecededBy(pp.Char('abc_$%^' + pp.nums)) + num
        very_boring_num = pp.PrecededBy(pp.WordStart()) + num
        finicky_num = pp.PrecededBy(pp.Word('^', '$%^'), retreat=3) + num
        s = 'c384 b8324 _9293874 _293 404 $%^$^%$2939'
        print_(s)
        for expr, expected_list, expected_dict in [(interesting_num, [384, 8324], {'prefix': ['c', 'b']}), (semi_interesting_num, [9293874, 293], {}), (boring_num, [404], {}), (crazy_num, [2939], {'prefix': ['^%$']}), (finicky_num, [2939], {}), (very_boring_num, [404], {})]:
            print_(expr.searchString(s))
            result = sum(expr.searchString(s))
            print_(result)
            self.assertEqual(result.asList(), expected_list, 'Erroneous tokens for {0}: expected {1}, got {2}'.format(expr, expected_list, result.asList()))
            self.assertEqual(result.asDict(), expected_dict, 'Erroneous named results for {0}: expected {1}, got {2}'.format(expr, expected_dict, result.asDict()))
        string_test = 'notworking'
        negs_pb = pp.PrecededBy('not', retreat=100)('negs_lb')
        pattern = pp.Group(negs_pb + pp.Literal('working'))('main')
        results = pattern.searchString(string_test)
        try:
            print_(results.dump())
        except RecursionError:
            self.assertTrue(False, 'got maximum excursion limit exception')
        else:
            self.assertTrue(True, 'got maximum excursion limit exception')