from collections import namedtuple
from functools import singledispatch as simplegeneric
import importlib
import importlib.util
import importlib.machinery
import os
import os.path
import sys
from types import ModuleType
import warnings
def _reopen(self):
    if self.file and self.file.closed:
        mod_type = self.etc[2]
        if mod_type == imp.PY_SOURCE:
            self.file = open(self.filename, 'r')
        elif mod_type in (imp.PY_COMPILED, imp.C_EXTENSION):
            self.file = open(self.filename, 'rb')