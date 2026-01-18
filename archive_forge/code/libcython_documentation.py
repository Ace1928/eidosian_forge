from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython

        Evaluate `code` in a Python or Cython stack frame using the given
        `input_type`.
        