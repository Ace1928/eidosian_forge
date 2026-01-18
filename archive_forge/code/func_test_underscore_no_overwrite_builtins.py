import sys
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.core.displayhook import CapturingDisplayHook
from IPython.utils.capture import CapturedIO
def test_underscore_no_overwrite_builtins():
    ip.run_cell("import gettext ; gettext.install('foo')", store_history=True)
    ip.run_cell('3+3', store_history=True)
    with AssertPrints('gettext'):
        ip.run_cell('print(_)', store_history=True)
    ip.run_cell('_ = "userset"', store_history=True)
    with AssertPrints('userset'):
        ip.run_cell('print(_)', store_history=True)
    ip.run_cell('import builtins; del builtins._')