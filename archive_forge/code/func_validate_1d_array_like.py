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
def validate_1d_array_like(func_name, seq):
    if isinstance(seq, types.Array):
        if seq.ndim != 1:
            raise TypeError('{0}(): input should have dimension 1'.format(func_name))
    elif not isinstance(seq, types.Sequence):
        raise TypeError('{0}(): input should be an array or sequence'.format(func_name))