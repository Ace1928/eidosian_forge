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
@overload(_system_check_non_empty)
def _system_check_non_empty_impl(a, b):
    ndim = b.ndim
    if ndim == 1:

        def oneD_impl(a, b):
            am = a.shape[-2]
            an = a.shape[-1]
            bm = b.shape[-1]
            if am == 0 or bm == 0 or an == 0:
                raise np.linalg.LinAlgError('Arrays cannot be empty')
        return oneD_impl
    else:

        def twoD_impl(a, b):
            am = a.shape[-2]
            an = a.shape[-1]
            bm = b.shape[-2]
            bn = b.shape[-1]
            if am == 0 or bm == 0 or an == 0 or (bn == 0):
                raise np.linalg.LinAlgError('Arrays cannot be empty')
        return twoD_impl