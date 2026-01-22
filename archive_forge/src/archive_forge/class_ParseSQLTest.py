from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseSQLTest(ParseTestCase):

    def runTest(self):
        import examples.simpleSQL as simpleSQL

        def test(s, numToks, errloc=-1):
            try:
                sqlToks = flatten(simpleSQL.simpleSQL.parseString(s).asList())
                print_(s, sqlToks, len(sqlToks))
                self.assertEqual(len(sqlToks), numToks, 'invalid parsed tokens, expected {0}, found {1} ({2})'.format(numToks, len(sqlToks), sqlToks))
            except ParseException as e:
                if errloc >= 0:
                    self.assertEqual(e.loc, errloc, 'expected error at {0}, found at {1}'.format(errloc, e.loc))
        test('SELECT * from XYZZY, ABC', 6)
        test('select * from SYS.XYZZY', 5)
        test('Select A from Sys.dual', 5)
        test('Select A,B,C from Sys.dual', 7)
        test('Select A, B, C from Sys.dual', 7)
        test('Select A, B, C from Sys.dual, Table2   ', 8)
        test('Xelect A, B, C from Sys.dual', 0, 0)
        test('Select A, B, C frox Sys.dual', 0, 15)
        test('Select', 0, 6)
        test('Select &&& frox Sys.dual', 0, 7)
        test("Select A from Sys.dual where a in ('RED','GREEN','BLUE')", 12)
        test("Select A from Sys.dual where a in ('RED','GREEN','BLUE') and b in (10,20,30)", 20)
        test('Select A,b from table1,table2 where table1.id eq table2.id -- test out comparison operators', 10)