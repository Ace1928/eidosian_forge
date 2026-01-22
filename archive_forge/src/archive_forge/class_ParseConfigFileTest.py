from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseConfigFileTest(ParseTestCase):

    def runTest(self):
        from examples import configParse

        def test(fnam, numToks, resCheckList):
            print_('Parsing', fnam, '...', end=' ')
            with open(fnam) as infile:
                iniFileLines = '\n'.join(infile.read().splitlines())
            iniData = configParse.inifile_BNF().parseString(iniFileLines)
            print_(len(flatten(iniData.asList())))
            print_(list(iniData.keys()))
            self.assertEqual(len(flatten(iniData.asList())), numToks, 'file %s not parsed correctly' % fnam)
            for chk in resCheckList:
                var = iniData
                for attr in chk[0].split('.'):
                    var = getattr(var, attr)
                print_(chk[0], var, chk[1])
                self.assertEqual(var, chk[1], 'ParseConfigFileTest: failed to parse ini {0!r} as expected {1}, found {2}'.format(chk[0], chk[1], var))
            print_('OK')
        test('test/karthik.ini', 23, [('users.K', '8'), ('users.mod_scheme', "'QPSK'"), ('users.Na', 'K+2')])
        test('examples/Setup.ini', 125, [('Startup.audioinf', 'M3i'), ('Languages.key1', '0x0003'), ('test.foo', 'bar')])