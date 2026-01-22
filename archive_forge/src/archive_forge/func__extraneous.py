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
def _extraneous(f: _T_Callable) -> _T_Callable:
    """
    Mark the given callable as extraneous to inlineCallbacks exception
    reporting; don't show these functions.

    @param f: a function that you NEVER WANT TO SEE AGAIN in ANY TRACEBACK
        reported by Failure.

    @type f: function

    @return: f
    """
    _inlineCallbacksExtraneous.append(f.__code__)
    return f