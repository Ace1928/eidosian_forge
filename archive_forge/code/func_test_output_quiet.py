import sys
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.core.displayhook import CapturingDisplayHook
from IPython.utils.capture import CapturedIO
def test_output_quiet():
    """Checking to make sure that output is quiet"""
    with AssertNotPrints('2'):
        ip.run_cell('1+1;', store_history=True)
    with AssertNotPrints('2'):
        ip.run_cell('1+1; # comment with a semicolon', store_history=True)
    with AssertNotPrints('2'):
        ip.run_cell('1+1;\n#commented_out_function()', store_history=True)