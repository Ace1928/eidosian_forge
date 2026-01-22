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
class _BLAS:
    """
    Functions to return type signatures for wrapped
    BLAS functions.
    """

    def __init__(self):
        ensure_blas()

    @classmethod
    def numba_xxnrm2(cls, dtype):
        rtype = getattr(dtype, 'underlying_float', dtype)
        sig = types.intc(types.char, types.intp, types.CPointer(dtype), types.intp, types.CPointer(rtype))
        return types.ExternalFunction('numba_xxnrm2', sig)

    @classmethod
    def numba_xxgemm(cls, dtype):
        sig = types.intc(types.char, types.char, types.char, types.intp, types.intp, types.intp, types.CPointer(dtype), types.CPointer(dtype), types.intp, types.CPointer(dtype), types.intp, types.CPointer(dtype), types.CPointer(dtype), types.intp)
        return types.ExternalFunction('numba_xxgemm', sig)