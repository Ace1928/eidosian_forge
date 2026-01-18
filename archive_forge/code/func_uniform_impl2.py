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
@overload(random.uniform)
def uniform_impl2(a, b):
    if isinstance(a, (types.Float, types.Integer)) and isinstance(b, (types.Float, types.Integer)):

        @intrinsic
        def _impl(typingcontext, a, b):
            low_preprocessor = _double_preprocessor(a)
            high_preprocessor = _double_preprocessor(b)
            return (signature(types.float64, a, b), uniform_impl('py', low_preprocessor, high_preprocessor))
        return lambda a, b: _impl(a, b)