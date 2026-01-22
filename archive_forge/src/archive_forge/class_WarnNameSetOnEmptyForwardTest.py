from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class WarnNameSetOnEmptyForwardTest(ParseTestCase):
    """
     - warn_name_set_on_empty_Forward - flag to enable warnings whan a Forward is defined
       with a results name, but has no contents defined (default=False)
    """

    def runTest(self):
        import pyparsing as pp
        pp.__diag__.warn_name_set_on_empty_Forward = True
        base = pp.Forward()
        if PY_3:
            with self.assertWarns(UserWarning, msg='failed to warn when naming an empty Forward expression'):
                base('x')