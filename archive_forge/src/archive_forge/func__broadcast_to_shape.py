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
def _broadcast_to_shape(context, builder, arrtype, arr, target_shape):
    """
    Broadcast the given array to the target_shape.
    Returns (array_type, array)
    """
    shapes = cgutils.unpack_tuple(builder, arr.shape)
    strides = cgutils.unpack_tuple(builder, arr.strides)
    shapes, strides = _bc_adjust_dimension(context, builder, shapes, strides, target_shape)
    shapes, strides = _bc_adjust_shape_strides(context, builder, shapes, strides, target_shape)
    new_arrtype = arrtype.copy(ndim=len(target_shape), layout='A')
    new_arr = make_array(new_arrtype)(context, builder)
    populate_array(new_arr, data=arr.data, shape=cgutils.pack_array(builder, shapes), strides=cgutils.pack_array(builder, strides), itemsize=arr.itemsize, meminfo=arr.meminfo, parent=arr.parent)
    return (new_arrtype, new_arr)