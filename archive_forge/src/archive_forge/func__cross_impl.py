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
@overload(_cross)
def _cross_impl(a, b):
    dtype = np.promote_types(as_dtype(a.dtype), as_dtype(b.dtype))
    if a.ndim == 1 and b.ndim == 1:

        def impl(a, b):
            cp = np.empty((3,), dtype)
            _cross_operation(a, b, cp)
            return cp
    else:

        def impl(a, b):
            shape = np.add(a[..., 0], b[..., 0]).shape
            cp = np.empty(shape + (3,), dtype)
            _cross_operation(a, b, cp)
            return cp
    return impl