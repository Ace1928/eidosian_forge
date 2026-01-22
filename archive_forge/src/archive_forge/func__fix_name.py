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
def _fix_name(self, fullname):
    if fullname is None:
        fullname = self.fullname
    elif fullname != self.fullname:
        raise ImportError('Loader for module %s cannot handle module %s' % (self.fullname, fullname))
    return fullname