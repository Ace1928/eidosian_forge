import gc
import signal
import sys
import textwrap
import weakref
import pytest
from pyarrow.util import doc, _break_traceback_cycle_from_frame
from pyarrow.tests.util import disabled_gc
def test_signal_refcycle():
    with disabled_gc():
        wr = exhibit_signal_refcycle()
        if wr() is None:
            pytest.skip("Python version does not have the bug we're testing for")
    gc.collect()
    with disabled_gc():
        wr = exhibit_signal_refcycle()
        assert wr() is not None
        _break_traceback_cycle_from_frame(sys._getframe(0))
        assert wr() is None