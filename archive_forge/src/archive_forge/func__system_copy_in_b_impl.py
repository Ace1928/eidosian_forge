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
@overload(_system_copy_in_b)
def _system_copy_in_b_impl(bcpy, b, nrhs):
    if b.ndim == 1:

        def oneD_impl(bcpy, b, nrhs):
            bcpy[:b.shape[-1], 0] = b
        return oneD_impl
    else:

        def twoD_impl(bcpy, b, nrhs):
            bcpy[:b.shape[-2], :nrhs] = b
        return twoD_impl