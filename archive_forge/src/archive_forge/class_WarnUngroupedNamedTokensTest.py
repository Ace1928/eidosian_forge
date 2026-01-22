from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class WarnUngroupedNamedTokensTest(ParseTestCase):
    """
     - warn_ungrouped_named_tokens_in_collection - flag to enable warnings when a results
       name is defined on a containing expression with ungrouped subexpressions that also
       have results names (default=True)
    """

    def runTest(self):
        import pyparsing as pp
        ppc = pp.pyparsing_common
        pp.__diag__.warn_ungrouped_named_tokens_in_collection = True
        COMMA = pp.Suppress(',').setName('comma')
        coord = ppc.integer('x') + COMMA + ppc.integer('y')
        if PY_3:
            with self.assertWarns(UserWarning, msg='failed to warn with named repetition of ungrouped named expressions'):
                path = coord[...].setResultsName('path')
        pp.__diag__.warn_ungrouped_named_tokens_in_collection = False