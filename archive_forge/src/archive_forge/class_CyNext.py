from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyNext(CyStep):
    """Step-over Cython, Python or C code."""
    name = 'cy -next'
    stepinto = False