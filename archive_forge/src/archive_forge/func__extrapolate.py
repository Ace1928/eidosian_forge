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
def _extrapolate(self, otherFailure):
    """
        Extrapolate from one failure into another, copying its stack frames.

        @param otherFailure: Another L{Failure}, whose traceback information,
            if any, should be preserved as part of the stack presented by this
            one.
        @type otherFailure: L{Failure}
        """
    self.__dict__ = copy.copy(otherFailure.__dict__)
    _, _, tb = sys.exc_info()
    frames = []
    while tb is not None:
        f = tb.tb_frame
        if f.f_code not in _inlineCallbacksExtraneous:
            frames.append((f.f_code.co_name, f.f_code.co_filename, tb.tb_lineno, (), ()))
        tb = tb.tb_next
    frames.extend(self.frames)
    self.frames = frames