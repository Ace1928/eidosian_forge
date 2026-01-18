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
@overload(random.weibullvariate)
def weibullvariate_impl(alpha, beta):
    if isinstance(alpha, (types.Float, types.Integer)) and isinstance(beta, (types.Float, types.Integer)):

        def _impl(alpha, beta):
            """Weibull distribution.  Taken from CPython."""
            u = 1.0 - random.random()
            return alpha * (-math.log(u)) ** (1.0 / beta)
        return _impl