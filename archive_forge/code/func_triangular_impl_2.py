import math
import random
import numpy as np
from llvmlite import ir
from numba.core.cgutils import is_nonelike
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core.imputils import (Registry, impl_ret_untracked,
from numba.core.typing import signature
from numba.core import types, cgutils
from numba.np import arrayobj
from numba.core.errors import NumbaTypeError
@overload(random.triangular)
def triangular_impl_2(low, high):

    def _impl(low, high):
        u = random.random()
        c = 0.5
        if u > c:
            u = 1.0 - u
            low, high = (high, low)
        return low + (high - low) * math.sqrt(u * c)
    if isinstance(low, (types.Float, types.Integer)) and isinstance(high, (types.Float, types.Integer)):
        return _impl