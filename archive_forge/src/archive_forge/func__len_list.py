import sys
import types as Types
import warnings
import weakref as Weakref
from inspect import isbuiltin, isclass, iscode, isframe, isfunction, ismethod, ismodule
from math import log
from os import curdir, linesep
from struct import calcsize
from gc import get_objects as _getobjects
from gc import get_referents as _getreferents  # containers only?
from array import array as _array  # array type
def _len_list(obj):
    """Length of list (estimate)."""
    n = len(obj)
    if n > 8:
        n += 6 + (n >> 3)
    elif n:
        n += 4
    return n