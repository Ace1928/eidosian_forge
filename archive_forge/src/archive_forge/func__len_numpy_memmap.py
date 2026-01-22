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
def _len_numpy_memmap(obj):
    """Approximate NumPy memmap in-memory size (in bytes!)."""
    nb = int(obj.nbytes * _amapped)
    return (nb + _PAGESIZE - 1) // _PAGESIZE * _PAGESIZE