import numpy as np
import operator
import llvmlite.ir
from llvmlite.ir import Constant
from numba.core import types, cgutils
from numba.core.cgutils import create_constant_array
from numba.core.imputils import (lower_builtin, lower_constant,
from numba.np import npdatetime_helpers, numpy_support, npyfuncs
from numba.extending import overload_method
from numba.core.config import IS_32BITS
from numba.core.errors import LoweringError
def _cast_to_timedelta(context, builder, val):
    temp = builder.alloca(TIMEDELTA64)
    val_is_nan = builder.fcmp_unordered('uno', val, val)
    with builder.if_else(val_is_nan) as (then, els):
        with then:
            builder.store(NAT, temp)
        with els:
            builder.store(builder.fptosi(val, TIMEDELTA64), temp)
    return builder.load(temp)