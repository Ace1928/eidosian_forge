import functools
import numbers
import sys
import numpy as np
from . import numerictypes as _nt
from .umath import absolute, isinf, isfinite, isnat
from . import multiarray
from .multiarray import (array, dragon4_positional, dragon4_scientific,
from .fromnumeric import any
from .numeric import concatenate, asarray, errstate
from .numerictypes import (longlong, intc, int_, float_, complex_, bool_,
from .overrides import array_function_dispatch, set_module
import operator
import warnings
import contextlib
def _leading_trailing(a, edgeitems, index=()):
    """
    Keep only the N-D corners (leading and trailing edges) of an array.

    Should be passed a base-class ndarray, since it makes no guarantees about
    preserving subclasses.
    """
    axis = len(index)
    if axis == a.ndim:
        return a[index]
    if a.shape[axis] > 2 * edgeitems:
        return concatenate((_leading_trailing(a, edgeitems, index + np.index_exp[:edgeitems]), _leading_trailing(a, edgeitems, index + np.index_exp[-edgeitems:])), axis=axis)
    else:
        return _leading_trailing(a, edgeitems, index + np.index_exp[:])