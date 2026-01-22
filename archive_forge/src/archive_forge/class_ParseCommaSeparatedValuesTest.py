from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseCommaSeparatedValuesTest(ParseTestCase):

    def runTest(self):
        from pyparsing import commaSeparatedList
        testData = ['a,b,c,100.2,,3', 'd, e, j k , m  ', "'Hello, World', f, g , , 5.1,x", 'John Doe, 123 Main St., Cleveland, Ohio', 'Jane Doe, 456 St. James St., Los Angeles , California   ', '']
        testVals = [[(3, '100.2'), (4, ''), (5, '3')], [(2, 'j k'), (3, 'm')], [(0, "'Hello, World'"), (2, 'g'), (3, '')], [(0, 'John Doe'), (1, '123 Main St.'), (2, 'Cleveland'), (3, 'Ohio')], [(0, 'Jane Doe'), (1, '456 St. James St.'), (2, 'Los Angeles'), (3, 'California')]]
        for line, tests in zip(testData, testVals):
            print_('Parsing: "' + line + '" ->', end=' ')
            results = commaSeparatedList.parseString(line)
            print_(results.asList())
            for t in tests:
                if not (len(results) > t[0] and results[t[0]] == t[1]):
                    print_('$$$', results.dump())
                    print_('$$$', results[0])
                self.assertTrue(len(results) > t[0] and results[t[0]] == t[1], "failed on %s, item %d s/b '%s', got '%s'" % (line, t[0], t[1], str(results.asList())))