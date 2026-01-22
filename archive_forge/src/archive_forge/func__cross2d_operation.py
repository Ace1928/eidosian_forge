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
def _cross2d_operation(a, b):

    def _cross_preprocessing(x):
        x0 = x[..., 0]
        x1 = x[..., 1]
        return (x0, x1)
    a0, a1 = _cross_preprocessing(a)
    b0, b1 = _cross_preprocessing(b)
    cp = np.multiply(a0, b1) - np.multiply(a1, b0)
    return np.asarray(cp)