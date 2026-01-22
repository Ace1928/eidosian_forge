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
def _insert_axis_in_strides(context, builder, orig_strides, ndim, axis):
    """
    Same as _insert_axis_in_shape(), but with a strides array.
    """
    assert len(orig_strides) == ndim - 1
    ll_shty = ir.ArrayType(cgutils.intp_t, ndim)
    strides = cgutils.alloca_once(builder, ll_shty)
    one = cgutils.intp_t(1)
    zero = cgutils.intp_t(0)
    for dim in range(ndim - 1):
        ll_dim = cgutils.intp_t(dim)
        after_axis = builder.icmp_signed('>=', ll_dim, axis)
        idx = builder.select(after_axis, builder.add(ll_dim, one), ll_dim)
        builder.store(orig_strides[dim], cgutils.gep_inbounds(builder, strides, 0, idx))
    builder.store(zero, cgutils.gep_inbounds(builder, strides, 0, axis))
    return cgutils.unpack_tuple(builder, builder.load(strides))