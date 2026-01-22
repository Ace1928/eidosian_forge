from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyFinish(CythonExecutionControlCommand):
    """
    Execute until the function returns.
    """
    name = 'cy finish'
    invoke = libpython.dont_suppress_errors(CythonExecutionControlCommand.finish)