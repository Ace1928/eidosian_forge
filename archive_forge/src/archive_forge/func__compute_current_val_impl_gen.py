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
def _compute_current_val_impl_gen(op, current_val, val):
    if isinstance(current_val, types.Complex):

        def impl(current_val, val):
            if op(val.real, current_val.real):
                return val
            elif val.real == current_val.real and op(val.imag, current_val.imag):
                return val
            return current_val
    else:

        def impl(current_val, val):
            return val if op(val, current_val) else current_val
    return impl