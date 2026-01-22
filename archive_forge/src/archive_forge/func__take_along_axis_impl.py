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
def _take_along_axis_impl(arr, indices, axis, Ni_orig, Nk_orig, indices_broadcast_shape):
    axis = normalize_axis('np.take_along_axis', 'axis', arr.ndim, axis)
    arr_shape = list(arr.shape)
    arr_shape[axis] = 1
    for i, (d1, d2) in enumerate(zip(arr_shape, indices.shape)):
        if d1 == 1:
            new_val = d2
        elif d2 == 1:
            new_val = d1
        else:
            if d1 != d2:
                raise ValueError("`arr` and `indices` dimensions don't match")
            new_val = d1
        indices_broadcast_shape = tuple_setitem(indices_broadcast_shape, i, new_val)
    arr_broadcast_shape = tuple_setitem(indices_broadcast_shape, axis, arr.shape[axis])
    arr = np.broadcast_to(arr, arr_broadcast_shape)
    indices = np.broadcast_to(indices, indices_broadcast_shape)
    Ni = Ni_orig
    if len(Ni_orig) > 0:
        for i in range(len(Ni)):
            Ni = tuple_setitem(Ni, i, arr.shape[i])
    Nk = Nk_orig
    if len(Nk_orig) > 0:
        for i in range(len(Nk)):
            Nk = tuple_setitem(Nk, i, arr.shape[axis + 1 + i])
    J = indices.shape[axis]
    out = np.empty(Ni + (J,) + Nk, arr.dtype)
    np_s_ = (slice(None, None, None),)
    for ii in np.ndindex(Ni):
        for kk in np.ndindex(Nk):
            a_1d = arr[ii + np_s_ + kk]
            indices_1d = indices[ii + np_s_ + kk]
            out_1d = out[ii + np_s_ + kk]
            for j in range(J):
                out_1d[j] = a_1d[indices_1d[j]]
    return out