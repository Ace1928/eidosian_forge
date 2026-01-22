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
@lower_getattr(types.Array, 'imag')
def array_imag_part(context, builder, typ, value):
    if typ.dtype in types.complex_domain:
        return array_complex_attr(context, builder, typ, value, attr='imag')
    elif typ.dtype in types.number_domain:
        sig = signature(typ.copy(readonly=True), typ)
        arrtype, shapes = _parse_empty_like_args(context, builder, sig, [value])
        ary = _empty_nd_impl(context, builder, arrtype, shapes)
        cgutils.memset(builder, ary.data, builder.mul(ary.itemsize, ary.nitems), 0)
        return impl_ret_new_ref(context, builder, sig.return_type, ary._getvalue())
    else:
        raise NotImplementedError('unsupported .imag for {}'.format(type.dtype))