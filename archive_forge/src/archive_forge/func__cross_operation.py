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
@register_jitable
def _cross_operation(a, b, out):

    def _cross_preprocessing(x):
        x0 = x[..., 0]
        x1 = x[..., 1]
        if x.shape[-1] == 3:
            x2 = x[..., 2]
        else:
            x2 = np.multiply(x.dtype.type(0), x0)
        return (x0, x1, x2)
    a0, a1, a2 = _cross_preprocessing(a)
    b0, b1, b2 = _cross_preprocessing(b)
    cp0 = np.multiply(a1, b2) - np.multiply(a2, b1)
    cp1 = np.multiply(a2, b0) - np.multiply(a0, b2)
    cp2 = np.multiply(a0, b1) - np.multiply(a1, b0)
    out[..., 0] = cp0
    out[..., 1] = cp1
    out[..., 2] = cp2