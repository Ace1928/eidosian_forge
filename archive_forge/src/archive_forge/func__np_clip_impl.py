import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
@register_jitable
def _np_clip_impl(a, a_min, a_max, out):
    ret = np.empty_like(a) if out is None else out
    a_b, a_min_b, a_max_b = np.broadcast_arrays(a, a_min, a_max)
    for index in np.ndindex(a_b.shape):
        val_a = a_b[index]
        val_a_min = a_min_b[index]
        val_a_max = a_max_b[index]
        ret[index] = min(max(val_a, val_a_min), val_a_max)
    return ret