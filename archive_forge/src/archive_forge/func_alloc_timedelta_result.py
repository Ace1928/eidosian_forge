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
def alloc_timedelta_result(builder, name='ret'):
    """
    Allocate a NaT-initialized datetime64 (or timedelta64) result slot.
    """
    ret = cgutils.alloca_once(builder, TIMEDELTA64, name=name)
    builder.store(NAT, ret)
    return ret