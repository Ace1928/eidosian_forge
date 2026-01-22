from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class TerminalBackground(CythonParameter):
    """
    Tell cygdb about the user's terminal background (light or dark).
    """