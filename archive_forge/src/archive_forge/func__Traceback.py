import builtins
import copy
import inspect
import linecache
import sys
from inspect import getmro
from io import StringIO
from typing import Callable, NoReturn, TypeVar
import opcode
from twisted.python import reflect
def _Traceback(stackFrames, tbFrames):
    """
    Construct a fake traceback object using a list of frames.

    It should have the same API as stdlib to allow interaction with
    other tools.

    @param stackFrames: [(methodname, filename, lineno, locals, globals), ...]
    @param tbFrames: [(methodname, filename, lineno, locals, globals), ...]
    """
    assert len(tbFrames) > 0, 'Must pass some frames'
    stack = None
    for sf in stackFrames:
        stack = _Frame(sf, stack)
    stack = _Frame(tbFrames[0], stack)
    firstTb = tb = _TracebackFrame(stack)
    for sf in tbFrames[1:]:
        stack = _Frame(sf, stack)
        tb.tb_next = _TracebackFrame(stack)
        tb = tb.tb_next
    return firstTb