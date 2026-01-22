from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseResultsNameBelowUngroupedNameTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        rule_num = pp.Regex('[0-9]+')('LIT_NUM*')
        list_num = pp.Group(pp.Literal('[')('START_LIST') + pp.delimitedList(rule_num)('LIST_VALUES') + pp.Literal(']')('END_LIST'))('LIST')
        test_string = '[ 1,2,3,4,5,6 ]'
        list_num.runTests(test_string)
        U = list_num.parseString(test_string)
        self.assertTrue('LIT_NUM' not in U.LIST.LIST_VALUES, 'results name retained as sub in ungrouped named result')