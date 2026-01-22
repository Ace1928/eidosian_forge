import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
def _same_kind(a, b):
    for t in [(types.CharSeq, types.Bytes), (types.UnicodeCharSeq, types.UnicodeType)]:
        if isinstance(a, t) and isinstance(b, t):
            return True
    return False