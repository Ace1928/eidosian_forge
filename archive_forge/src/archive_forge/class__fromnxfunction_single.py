import itertools
import warnings
from . import core as ma
from .core import (
import numpy as np
from numpy import ndarray, array as nxarray
from numpy.core.multiarray import normalize_axis_index
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import _ureduce
from numpy.lib.index_tricks import AxisConcatenator
class _fromnxfunction_single(_fromnxfunction):
    """
    A version of `_fromnxfunction` that is called with a single array
    argument followed by auxiliary args that are passed verbatim for
    both the data and mask calls.
    """

    def __call__(self, x, *args, **params):
        func = getattr(np, self.__name__)
        if isinstance(x, ndarray):
            _d = func(x.__array__(), *args, **params)
            _m = func(getmaskarray(x), *args, **params)
            return masked_array(_d, mask=_m)
        else:
            _d = func(np.asarray(x), *args, **params)
            _m = func(getmaskarray(x), *args, **params)
            return masked_array(_d, mask=_m)