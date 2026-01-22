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
@intrinsic
def _numpy_broadcast_to(typingctx, array, shape):
    ret = array.copy(ndim=shape.count, layout='A', readonly=True)
    sig = ret(array, shape)

    def codegen(context, builder, sig, args):
        src, shape_ = args
        srcty = sig.args[0]
        src = make_array(srcty)(context, builder, src)
        shape_ = cgutils.unpack_tuple(builder, shape_)
        _, dest = _broadcast_to_shape(context, builder, srcty, src, shape_)
        setattr(dest, 'parent', Constant(context.get_value_type(dest._datamodel.get_type('parent')), None))
        res = dest._getvalue()
        return impl_ret_borrowed(context, builder, sig.return_type, res)
    return (sig, codegen)