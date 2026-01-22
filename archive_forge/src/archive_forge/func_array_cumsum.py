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
@overload(np.cumsum)
@overload_method(types.Array, 'cumsum')
def array_cumsum(a):
    if isinstance(a, types.Array):
        is_integer = a.dtype in types.signed_domain
        is_bool = a.dtype == types.bool_
        if is_integer and a.dtype.bitwidth < types.intp.bitwidth or is_bool:
            dtype = as_dtype(types.intp)
        else:
            dtype = as_dtype(a.dtype)
        acc_init = get_accumulator(dtype, 0)

        def array_cumsum_impl(a):
            out = np.empty(a.size, dtype)
            c = acc_init
            for idx, v in enumerate(a.flat):
                c += v
                out[idx] = c
            return out
        return array_cumsum_impl