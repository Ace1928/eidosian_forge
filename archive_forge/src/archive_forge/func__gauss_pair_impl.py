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
def _gauss_pair_impl(_random):

    def compute_gauss_pair():
        """
        Compute a pair of numbers on the normal distribution.
        """
        while True:
            x1 = 2.0 * _random() - 1.0
            x2 = 2.0 * _random() - 1.0
            r2 = x1 * x1 + x2 * x2
            if r2 < 1.0 and r2 != 0.0:
                break
        f = math.sqrt(-2.0 * math.log(r2) / r2)
        return (f * x1, f * x2)
    return compute_gauss_pair