from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseFileTest(ParseTestCase):

    def runTest(self):
        from pyparsing import pyparsing_common, OneOrMore
        s = '\n        123 456 789\n        '
        input_file = StringIO(s)
        integer = pyparsing_common.integer
        results = OneOrMore(integer).parseFile(input_file)
        print_(results)
        results = OneOrMore(integer).parseFile('test/parsefiletest_input_file.txt')
        print_(results)