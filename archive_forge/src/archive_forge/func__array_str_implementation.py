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
def _array_str_implementation(a, max_line_width=None, precision=None, suppress_small=None, array2string=array2string):
    """Internal version of array_str() that allows overriding array2string."""
    if _format_options['legacy'] <= 113 and a.shape == () and (not a.dtype.names):
        return str(a.item())
    if a.shape == ():
        return _guarded_repr_or_str(np.ndarray.__getitem__(a, ()))
    return array2string(a, max_line_width, precision, suppress_small, ' ', '')