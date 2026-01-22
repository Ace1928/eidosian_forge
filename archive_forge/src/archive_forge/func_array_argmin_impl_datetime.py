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
def array_argmin_impl_datetime(arry):
    if arry.size == 0:
        raise ValueError('attempt to get argmin of an empty sequence')
    it = np.nditer(arry)
    min_value = next(it).take(0)
    min_idx = 0
    if np.isnat(min_value):
        return min_idx
    idx = 1
    for view in it:
        v = view.item()
        if np.isnat(v):
            return idx
        if v < min_value:
            min_value = v
            min_idx = idx
        idx += 1
    return min_idx