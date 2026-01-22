from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CompleteUnqualifiedFunctionNames(CythonParameter):
    """
    Have 'cy break' complete unqualified function or method names.
    """