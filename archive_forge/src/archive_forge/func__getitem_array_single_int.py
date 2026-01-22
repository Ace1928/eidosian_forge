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
def _getitem_array_single_int(context, builder, return_type, aryty, ary, idx):
    """ Evaluate `ary[idx]`, where idx is a single int. """
    shapes = cgutils.unpack_tuple(builder, ary.shape, count=aryty.ndim)
    strides = cgutils.unpack_tuple(builder, ary.strides, count=aryty.ndim)
    offset = builder.mul(strides[0], idx)
    dataptr = cgutils.pointer_add(builder, ary.data, offset)
    view_shapes = shapes[1:]
    view_strides = strides[1:]
    if isinstance(return_type, types.Buffer):
        retary = make_view(context, builder, aryty, ary, return_type, dataptr, view_shapes, view_strides)
        return retary._getvalue()
    else:
        assert not view_shapes
        return load_item(context, builder, aryty, dataptr)