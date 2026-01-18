import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def ufunc_can_cast(from_, to, has_mixed_inputs, casting='safe'):
    """
    A variant of np.can_cast() that can allow casting any integer to
    any real or complex type, in case the operation has mixed-kind
    inputs.

    For example we want `np.power(float32, int32)` to be computed using
    SP arithmetic and return `float32`.
    However, `np.sqrt(int32)` should use DP arithmetic and return `float64`.
    """
    from_ = np.dtype(from_)
    to = np.dtype(to)
    if has_mixed_inputs and from_.kind in 'iu' and (to.kind in 'cf'):
        return True
    return np.can_cast(from_, to, casting)