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
@register_jitable
def _handle_err_maybe_convergence_problem(r):
    if r != 0:
        if r < 0:
            fatal_error_func()
            assert 0
        if r > 0:
            raise ValueError('Internal algorithm failed to converge.')