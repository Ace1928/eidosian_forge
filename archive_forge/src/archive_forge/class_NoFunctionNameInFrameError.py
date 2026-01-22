from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class NoFunctionNameInFrameError(NoCythonFunctionInFrameError):
    """
    raised when the name of the C function could not be determined
    in the current C stack frame
    """
    msg = 'C function name could not be determined in the current C stack frame'