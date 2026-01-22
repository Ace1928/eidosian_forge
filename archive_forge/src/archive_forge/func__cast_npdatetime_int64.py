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
@lower_cast(types.NPDatetime, types.Integer)
@lower_cast(types.NPTimedelta, types.Integer)
def _cast_npdatetime_int64(context, builder, fromty, toty, val):
    if toty.bitwidth != 64:
        msg = f'Cannot cast {fromty} to {toty} as {toty} is not 64 bits wide.'
        raise ValueError(msg)
    return val