import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def walk_tb(tb):
    """Walk a traceback yielding the frame and line number for each frame.

    This will follow tb.tb_next (and thus is in the opposite order to
    walk_stack). Usually used with StackSummary.extract.
    """
    while tb is not None:
        yield (tb.tb_frame, tb.tb_lineno)
        tb = tb.tb_next