from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CythonVariable(object):

    def __init__(self, name, cname, qualified_name, type, lineno):
        self.name = name
        self.cname = cname
        self.qualified_name = qualified_name
        self.type = type
        self.lineno = int(lineno)