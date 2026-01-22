from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class SkipToParserTests(ParseTestCase):

    def runTest(self):
        from pyparsing import Literal, SkipTo, cStyleComment, ParseBaseException, And, Word, alphas, nums, Optional, NotAny
        thingToFind = Literal('working')
        testExpr = SkipTo(Literal(';'), include=True, ignore=cStyleComment) + thingToFind

        def tryToParse(someText, fail_expected=False):
            try:
                print_(testExpr.parseString(someText))
                self.assertFalse(fail_expected, 'expected failure but no exception raised')
            except Exception as e:
                print_('Exception %s while parsing string %s' % (e, repr(someText)))
                self.assertTrue(fail_expected and isinstance(e, ParseBaseException), 'Exception %s while parsing string %s' % (e, repr(someText)))
        tryToParse('some text /* comment with ; in */; working')
        tryToParse('some text /* comment with ; in */some other stuff; working')
        testExpr = SkipTo(Literal(';'), include=True, ignore=cStyleComment, failOn='other') + thingToFind
        tryToParse('some text /* comment with ; in */; working')
        tryToParse('some text /* comment with ; in */some other stuff; working', fail_expected=True)
        text = 'prefixDATAsuffix'
        data = Literal('DATA')
        suffix = Literal('suffix')
        expr = SkipTo(data + suffix)('prefix') + data + suffix
        result = expr.parseString(text)
        self.assertTrue(isinstance(result.prefix, str), 'SkipTo created with wrong saveAsList attribute')
        if PY_3:

            def define_expr(s):
                from pyparsing import Literal, And, Word, alphas, nums, Optional, NotAny
                alpha_word = (~Literal('end') + Word(alphas, asKeyword=True)).setName('alpha')
                num_word = Word(nums, asKeyword=True).setName('int')
                ret = eval(s)
                ret.streamline()
                print_(ret)
                return ret

            def test(expr, test_string, expected_list, expected_dict):
                try:
                    result = expr.parseString(test_string)
                except Exception as pe:
                    if any((expected is not None for expected in (expected_list, expected_dict))):
                        self.assertTrue(False, '{} failed to parse {!r}'.format(expr, test_string))
                else:
                    self.assertEqual(result.asList(), expected_list)
                    self.assertEqual(result.asDict(), expected_dict)
            e = define_expr('... + Literal("end")')
            test(e, 'start 123 end', ['start 123 ', 'end'], {'_skipped': ['start 123 ']})
            e = define_expr('Literal("start") + ... + Literal("end")')
            test(e, 'start 123 end', ['start', '123 ', 'end'], {'_skipped': ['123 ']})
            e = define_expr('Literal("start") + ...')
            test(e, 'start 123 end', None, None)
            e = define_expr('And(["start", ..., "end"])')
            test(e, 'start 123 end', ['start', '123 ', 'end'], {'_skipped': ['123 ']})
            e = define_expr('And([..., "end"])')
            test(e, 'start 123 end', ['start 123 ', 'end'], {'_skipped': ['start 123 ']})
            e = define_expr('"start" + (num_word | ...) + "end"')
            test(e, 'start 456 end', ['start', '456', 'end'], {})
            test(e, 'start 123 456 end', ['start', '123', '456 ', 'end'], {'_skipped': ['456 ']})
            test(e, 'start end', ['start', '', 'end'], {'_skipped': ['missing <int>']})
            e = define_expr('"start" + (alpha_word[...] & num_word[...] | ...) + "end"')
            test(e, 'start 456 red end', ['start', '456', 'red', 'end'], {})
            test(e, 'start red 456 end', ['start', 'red', '456', 'end'], {})
            test(e, 'start 456 red + end', ['start', '456', 'red', '+ ', 'end'], {'_skipped': ['+ ']})
            test(e, 'start red end', ['start', 'red', 'end'], {})
            test(e, 'start 456 end', ['start', '456', 'end'], {})
            test(e, 'start end', ['start', 'end'], {})
            test(e, 'start 456 + end', ['start', '456', '+ ', 'end'], {'_skipped': ['+ ']})
            e = define_expr('"start" + (alpha_word[1, ...] & num_word[1, ...] | ...) + "end"')
            test(e, 'start 456 red end', ['start', '456', 'red', 'end'], {})
            test(e, 'start red 456 end', ['start', 'red', '456', 'end'], {})
            test(e, 'start 456 red + end', ['start', '456', 'red', '+ ', 'end'], {'_skipped': ['+ ']})
            test(e, 'start red end', ['start', 'red ', 'end'], {'_skipped': ['red ']})
            test(e, 'start 456 end', ['start', '456 ', 'end'], {'_skipped': ['456 ']})
            test(e, 'start end', ['start', '', 'end'], {'_skipped': ['missing <{{alpha}... & {int}...}>']})
            test(e, 'start 456 + end', ['start', '456 + ', 'end'], {'_skipped': ['456 + ']})
            e = define_expr('"start" + (alpha_word | ...) + (num_word | ...) + "end"')
            test(e, 'start red 456 end', ['start', 'red', '456', 'end'], {})
            test(e, 'start red end', ['start', 'red', '', 'end'], {'_skipped': ['missing <int>']})
            test(e, 'start end', ['start', '', '', 'end'], {'_skipped': ['missing <alpha>', 'missing <int>']})
            e = define_expr('Literal("start") + ... + "+" + ... + "end"')
            test(e, 'start red + 456 end', ['start', 'red ', '+', '456 ', 'end'], {'_skipped': ['red ', '456 ']})