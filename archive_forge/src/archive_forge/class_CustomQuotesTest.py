from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class CustomQuotesTest(ParseTestCase):

    def runTest(self):
        self.expect_warning = True
        from pyparsing import QuotedString
        testString = '\n            sdlfjs :sdf\\:jls::djf: sl:kfsjf\n            sdlfjs -sdf\\:jls::--djf: sl-kfsjf\n            sdlfjs -sdf\\:::jls::--djf: sl:::-kfsjf\n            sdlfjs ^sdf\\:jls^^--djf^ sl-kfsjf\n            sdlfjs ^^^==sdf\\:j=lz::--djf: sl=^^=kfsjf\n            sdlfjs ==sdf\\:j=ls::--djf: sl==kfsjf^^^\n        '
        colonQuotes = QuotedString(':', '\\', '::')
        dashQuotes = QuotedString('-', '\\', '--')
        hatQuotes = QuotedString('^', '\\')
        hatQuotes1 = QuotedString('^', '\\', '^^')
        dblEqQuotes = QuotedString('==', '\\')

        def test(quoteExpr, expected):
            print_(quoteExpr.pattern)
            print_(quoteExpr.searchString(testString))
            print_(quoteExpr.searchString(testString)[0][0])
            print_(expected)
            self.assertEqual(quoteExpr.searchString(testString)[0][0], expected, "failed to match %s, expected '%s', got '%s'" % (quoteExpr, expected, quoteExpr.searchString(testString)[0]))
            print_()
        test(colonQuotes, 'sdf:jls:djf')
        test(dashQuotes, 'sdf:jls::-djf: sl')
        test(hatQuotes, 'sdf:jls')
        test(hatQuotes1, 'sdf:jls^--djf')
        test(dblEqQuotes, 'sdf:j=ls::--djf: sl')
        test(QuotedString(':::'), 'jls::--djf: sl')
        test(QuotedString('==', endQuoteChar='--'), 'sdf\\:j=lz::')
        test(QuotedString('^^^', multiline=True), '==sdf\\:j=lz::--djf: sl=^^=kfsjf\n            sdlfjs ==sdf\\:j=ls::--djf: sl==kfsjf')
        try:
            bad1 = QuotedString('', '\\')
        except SyntaxError as se:
            pass
        else:
            self.assertTrue(False, 'failed to raise SyntaxError with empty quote string')