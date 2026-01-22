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
@intrinsic
def _gen_index_tuple(tyctx, shape_tuple, value, axis):
    """
    Generates a tuple that can be used to index a specific slice from an
    array for sum with axis.  shape_tuple is the size of the dimensions of
    the input array.  'value' is the value to put in the indexing tuple
    in the axis dimension and 'axis' is that dimension.  For this to work,
    axis has to be a const.
    """
    if not isinstance(axis, types.Literal):
        raise RequireLiteralValue('axis argument must be a constant')
    axis_value = axis.literal_value
    nd = len(shape_tuple)
    if axis_value >= nd:
        axis_value = 0
    before = axis_value
    after = nd - before - 1
    types_list = []
    types_list += [types.slice2_type] * before
    types_list += [types.intp]
    types_list += [types.slice2_type] * after
    tupty = types.Tuple(types_list)
    function_sig = tupty(shape_tuple, value, axis)

    def codegen(cgctx, builder, signature, args):
        lltupty = cgctx.get_value_type(tupty)
        tup = cgutils.get_null_value(lltupty)
        [_, value_arg, _] = args

        def create_full_slice():
            return slice(None, None)
        slice_data = cgctx.compile_internal(builder, create_full_slice, types.slice2_type(), [])
        for i in range(0, axis_value):
            tup = builder.insert_value(tup, slice_data, i)
        tup = builder.insert_value(tup, value_arg, axis_value)
        for i in range(axis_value + 1, nd):
            tup = builder.insert_value(tup, slice_data, i)
        return tup
    return (function_sig, codegen)