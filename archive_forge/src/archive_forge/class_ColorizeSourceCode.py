from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class ColorizeSourceCode(CythonParameter):
    """
    Tell cygdb whether to colorize source code.
    """