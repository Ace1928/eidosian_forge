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
@overload(np.mean)
@overload_method(types.Array, 'mean')
def array_mean(a):
    if isinstance(a, types.Array):
        is_number = a.dtype in types.integer_domain | frozenset([types.bool_])
        if is_number:
            dtype = as_dtype(types.float64)
        else:
            dtype = as_dtype(a.dtype)
        acc_init = get_accumulator(dtype, 0)

        def array_mean_impl(a):
            c = acc_init
            for v in np.nditer(a):
                c += v.item()
            return c / a.size
        return array_mean_impl