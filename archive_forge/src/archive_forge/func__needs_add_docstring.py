import functools
import warnings
import operator
import types
import numpy as np
from . import numeric as _nx
from .numeric import result_type, NaN, asanyarray, ndim
from numpy.core.multiarray import add_docstring
from numpy.core import overrides
def _needs_add_docstring(obj):
    """
    Returns true if the only way to set the docstring of `obj` from python is
    via add_docstring.

    This function errs on the side of being overly conservative.
    """
    Py_TPFLAGS_HEAPTYPE = 1 << 9
    if isinstance(obj, (types.FunctionType, types.MethodType, property)):
        return False
    if isinstance(obj, type) and obj.__flags__ & Py_TPFLAGS_HEAPTYPE:
        return False
    return True