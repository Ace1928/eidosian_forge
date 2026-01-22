from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class RequiredEachTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Keyword
        parser = Keyword('bam') & Keyword('boo')
        try:
            res1 = parser.parseString('bam boo')
            print_(res1.asList())
            res2 = parser.parseString('boo bam')
            print_(res2.asList())
        except ParseException:
            failed = True
        else:
            failed = False
            self.assertFalse(failed, 'invalid logic in Each')
            self.assertEqual(set(res1), set(res2), 'Failed RequiredEachTest, expected ' + str(res1.asList()) + ' and ' + str(res2.asList()) + 'to contain same words in any order')