from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class SetBreakTest(ParseTestCase):
    """
    Test behavior of ParserElement.setBreak(), to invoke the debugger before parsing that element is attempted.

    Temporarily monkeypatches pdb.set_trace.
    """

    def runTest(self):
        was_called = []

        def mock_set_trace():
            was_called.append(True)
        import pyparsing as pp
        wd = pp.Word(pp.alphas)
        wd.setBreak()
        print_('Before parsing with setBreak:', was_called)
        import pdb
        with AutoReset(pdb, 'set_trace'):
            pdb.set_trace = mock_set_trace
            wd.parseString('ABC')
        print_('After parsing with setBreak:', was_called)
        self.assertTrue(bool(was_called), "set_trace wasn't called by setBreak")