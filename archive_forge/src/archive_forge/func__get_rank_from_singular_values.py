import contextlib
import warnings
from llvmlite import ir
import numpy as np
import operator
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError, \
from .arrayobj import make_array, _empty_nd_impl, array_copy
from numba.np import numpy_support as np_support
@register_jitable
def _get_rank_from_singular_values(sv, t):
    """
    Gets rank from singular values with cut-off at a given tolerance
    """
    rank = 0
    for k in range(len(sv)):
        if sv[k] > t:
            rank = rank + 1
        else:
            break
    return rank