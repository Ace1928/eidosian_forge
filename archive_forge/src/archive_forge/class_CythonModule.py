from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CythonModule(object):

    def __init__(self, module_name, filename, c_filename):
        self.name = module_name
        self.filename = filename
        self.c_filename = c_filename
        self.globals = {}
        self.lineno_cy2c = {}
        self.lineno_c2cy = {}
        self.functions = {}