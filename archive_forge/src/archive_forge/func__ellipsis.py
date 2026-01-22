from __future__ import absolute_import, division, print_function
import ctypes
from itertools import chain
from . import coretypes as ct
def _ellipsis(name):
    return ct.Ellipsis(ct.TypeVar(name))