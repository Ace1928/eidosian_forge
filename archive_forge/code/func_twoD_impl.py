import contextlib
import warnings
from llvmlite import ir
import numpy as np
import operator
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError, \
from .arrayobj import make_array, _empty_nd_impl, array_copy
from numba.np import numpy_support as np_support
def twoD_impl(x, ord=None):
    n = x.shape[-1]
    m = x.shape[-2]
    if x.size == 0:
        return 0.0
    if ord == np.inf:
        global_max = 0.0
        for ii in range(m):
            tmp = 0.0
            for jj in range(n):
                tmp += abs(x[ii, jj])
            if tmp > global_max:
                global_max = tmp
        return global_max
    elif ord == -np.inf:
        global_min = max_val
        for ii in range(m):
            tmp = 0.0
            for jj in range(n):
                tmp += abs(x[ii, jj])
            if tmp < global_min:
                global_min = tmp
        return global_min
    elif ord == 1:
        global_max = 0.0
        for ii in range(n):
            tmp = 0.0
            for jj in range(m):
                tmp += abs(x[jj, ii])
            if tmp > global_max:
                global_max = tmp
        return global_max
    elif ord == -1:
        global_min = max_val
        for ii in range(n):
            tmp = 0.0
            for jj in range(m):
                tmp += abs(x[jj, ii])
            if tmp < global_min:
                global_min = tmp
        return global_min
    elif ord == 2:
        return _compute_singular_values(x)[0]
    elif ord == -2:
        return _compute_singular_values(x)[-1]
    else:
        raise ValueError('Invalid norm order for matrices.')