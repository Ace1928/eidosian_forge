import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload_method(types.UnicodeCharSeq, 'zfill')
@overload_method(types.CharSeq, 'zfill')
@overload_method(types.Bytes, 'zfill')
def unicode_charseq_zfill(a, width):
    if isinstance(a, types.UnicodeCharSeq):

        def impl(a, width):
            return str(a).zfill(width)
        return impl
    if isinstance(a, (types.CharSeq, types.Bytes)):

        def impl(a, width):
            return a._to_str().zfill(width)._to_bytes()
        return impl