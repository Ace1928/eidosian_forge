import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload_method(types.UnicodeType, '_to_bytes')
def unicode_to_bytes_mth(s):
    """Convert unicode_type object to Bytes object.

    Note: The usage of _to_bytes method can be eliminated once all
    Python bytes operations are implemented for numba Bytes objects.

    """

    def impl(s):
        return _unicode_to_bytes(s)
    return impl