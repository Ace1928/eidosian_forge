from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class CloseMatchTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        searchseq = pp.CloseMatch('ATCATCGAATGGA', 2)
        _, results = searchseq.runTests('\n            ATCATCGAATGGA\n            XTCATCGAATGGX\n            ATCATCGAAXGGA\n            ATCAXXGAATGGA\n            ATCAXXGAATGXA\n            ATCAXXGAATGG\n            ')
        expected = ([], [0, 12], [9], [4, 5], None, None)
        for r, exp in zip(results, expected):
            if exp is not None:
                self.assertEqual(r[1].mismatches, exp, 'fail CloseMatch between %r and %r' % (searchseq.match_string, r[0]))
            print_(r[0], 'exc: %s' % r[1] if exp is None and isinstance(r[1], Exception) else ('no match', 'match')[r[1].mismatches == exp])