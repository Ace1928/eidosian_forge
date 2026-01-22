from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class EllipsisRepetionTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        import re
        word = pp.Word(pp.alphas).setName('word')
        num = pp.Word(pp.nums).setName('num')
        exprs = [word[...] + num, word[0, ...] + num, word[1, ...] + num, word[2, ...] + num, word[..., 3] + num, word[2] + num]
        expected_res = ['([abcd]+ )*\\d+', '([abcd]+ )*\\d+', '([abcd]+ )+\\d+', '([abcd]+ ){2,}\\d+', '([abcd]+ ){0,3}\\d+', '([abcd]+ ){2}\\d+']
        tests = ['aa bb cc dd 123', 'bb cc dd 123', 'cc dd 123', 'dd 123', '123']
        all_success = True
        for expr, expected_re in zip(exprs, expected_res):
            successful_tests = [t for t in tests if re.match(expected_re, t)]
            failure_tests = [t for t in tests if not re.match(expected_re, t)]
            success1, _ = expr.runTests(successful_tests)
            success2, _ = expr.runTests(failure_tests, failureTests=True)
            all_success = all_success and success1 and success2
            if not all_success:
                print_('Failed expression:', expr)
                break
        self.assertTrue(all_success, 'failed getItem_ellipsis test')