from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyDown(CyUp):
    """
    Go down a Cython, Python or relevant C frame.
    """
    name = 'cy down'
    _command = 'down'