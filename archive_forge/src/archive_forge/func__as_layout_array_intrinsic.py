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
def _as_layout_array_intrinsic(typingctx, a, output_layout):
    if not isinstance(output_layout, types.StringLiteral):
        raise errors.RequireLiteralValue(output_layout)
    ret = a.copy(layout=output_layout.literal_value, ndim=max(a.ndim, 1))
    sig = ret(a, output_layout)
    return (sig, lambda c, b, s, a: _as_layout_array(c, b, s, a, output_layout=output_layout.literal_value))