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
@overload(np.take_along_axis)
def arr_take_along_axis(arr, indices, axis):
    if not isinstance(arr, types.Array):
        raise errors.TypingError('The first argument "arr" must be an array')
    if not isinstance(indices, types.Array):
        raise errors.TypingError('The second argument "indices" must be an array')
    if not isinstance(indices.dtype, types.Integer):
        raise errors.TypingError('The indices array must contain integers')
    if is_nonelike(axis):
        arr_ndim = 1
    else:
        arr_ndim = arr.ndim
    if arr_ndim != indices.ndim:
        raise errors.TypingError('`indices` and `arr` must have the same number of dimensions')
    indices_broadcast_shape = tuple(range(indices.ndim))
    if is_nonelike(axis):

        def take_along_axis_impl(arr, indices, axis):
            return _take_along_axis_impl(arr.flatten(), indices, 0, (), (), indices_broadcast_shape)
    else:
        check_is_integer(axis, 'axis')
        if not isinstance(axis, types.IntegerLiteral):
            raise errors.NumbaValueError('axis must be a literal value')
        axis = axis.literal_value
        if axis < 0:
            axis = arr.ndim + axis
        if axis < 0 or axis >= arr.ndim:
            raise errors.NumbaValueError('axis is out of bounds')
        Ni = tuple(range(axis))
        Nk = tuple(range(axis + 1, arr.ndim))

        def take_along_axis_impl(arr, indices, axis):
            return _take_along_axis_impl(arr, indices, axis, Ni, Nk, indices_broadcast_shape)
    return take_along_axis_impl