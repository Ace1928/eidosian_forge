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
def _where_generic_impl(dtype, layout):
    use_faster_impl = layout in [{'C'}, {'F'}]

    def impl(condition, x, y):
        cond1, x1, y1 = (np.asarray(condition), np.asarray(x), np.asarray(y))
        shape = np.broadcast_shapes(cond1.shape, x1.shape, y1.shape)
        cond_ = np.broadcast_to(cond1, shape)
        x_ = np.broadcast_to(x1, shape)
        y_ = np.broadcast_to(y1, shape)
        if layout == 'F':
            res = np.empty(shape[::-1], dtype=dtype).T
        else:
            res = np.empty(shape, dtype=dtype)
        if use_faster_impl:
            return _where_fast_inner_impl(cond_, x_, y_, res)
        else:
            return _where_generic_inner_impl(cond_, x_, y_, res)
    return impl