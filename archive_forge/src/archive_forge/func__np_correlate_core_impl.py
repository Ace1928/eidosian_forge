import math
from collections import namedtuple
import operator
import warnings
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.extending import overload, overload_method, register_jitable
from numba.np.numpy_support import (as_dtype, type_can_asarray, type_is_scalar,
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.np.arrayobj import (make_array, load_item, store_item,
from numba.np.linalg import ensure_blas
from numba.core.extending import intrinsic
from numba.core.errors import (RequireLiteralValue, TypingError,
from numba.cpython.unsafe.tuple import tuple_setitem
@overload(_np_correlate_core)
def _np_correlate_core_impl(ap1, ap2, mode, direction):
    a_dt = as_dtype(ap1.dtype)
    b_dt = as_dtype(ap2.dtype)
    dt = np.promote_types(a_dt, b_dt)
    innerprod = _get_inner_prod(ap1.dtype, ap2.dtype)

    def impl(ap1, ap2, mode, direction):
        n1 = len(ap1)
        n2 = len(ap2)
        if n1 < n2:
            raise ValueError("'len(ap1)' must greater than 'len(ap2)'")
        length = n1
        n = n2
        if mode == 'valid':
            length = length - n + 1
            n_left = 0
            n_right = 0
        elif mode == 'full':
            n_right = n - 1
            n_left = n - 1
            length = length + n - 1
        elif mode == 'same':
            n_left = n // 2
            n_right = n - n_left - 1
        else:
            raise ValueError("Invalid 'mode', valid are 'full', 'same', 'valid'")
        ret = np.zeros(length, dt)
        if direction == 1:
            idx = 0
            inc = 1
        elif direction == -1:
            idx = length - 1
            inc = -1
        else:
            raise ValueError('Invalid direction')
        for i in range(n_left):
            k = i + n - n_left
            ret[idx] = innerprod(ap1[:k], ap2[-k:])
            idx = idx + inc
        for i in range(n1 - n2 + 1):
            ret[idx] = innerprod(ap1[i:i + n2], ap2)
            idx = idx + inc
        for i in range(n_right):
            k = n - i - 1
            ret[idx] = innerprod(ap1[-k:], ap2[:k])
            idx = idx + inc
        return ret
    return impl