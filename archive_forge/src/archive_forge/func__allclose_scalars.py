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
def _allclose_scalars(a_v, b_v, rtol=1e-05, atol=1e-08, equal_nan=False):
    a_v_isnan = np.isnan(a_v)
    b_v_isnan = np.isnan(b_v)
    if not a_v_isnan and b_v_isnan or (a_v_isnan and (not b_v_isnan)):
        return False
    if a_v_isnan and b_v_isnan:
        if not equal_nan:
            return False
    else:
        if np.isinf(a_v) or np.isinf(b_v):
            return a_v == b_v
        if np.abs(a_v - b_v) > atol + rtol * np.abs(b_v * 1.0):
            return False
    return True