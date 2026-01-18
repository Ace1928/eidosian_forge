import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def walk_stack(f):
    """Walk a stack yielding the frame and line number for each frame.

    This will follow f.f_back from the given frame. If no frame is given, the
    current stack is used. Usually used with StackSummary.extract.
    """
    if f is None:
        f = sys._getframe().f_back.f_back.f_back.f_back
    while f is not None:
        yield (f, f.f_lineno)
        f = f.f_back