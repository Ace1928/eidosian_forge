import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload_method(types.UnicodeCharSeq, 'startswith')
@overload_method(types.CharSeq, 'startswith')
@overload_method(types.Bytes, 'startswith')
def unicode_charseq_startswith(a, b):
    if isinstance(a, types.UnicodeCharSeq):
        if isinstance(b, types.UnicodeCharSeq):

            def impl(a, b):
                return str(a).startswith(str(b))
            return impl
        if isinstance(b, types.UnicodeType):

            def impl(a, b):
                return str(a).startswith(b)
            return impl
    if isinstance(a, (types.CharSeq, types.Bytes)):
        if isinstance(b, (types.CharSeq, types.Bytes)):

            def impl(a, b):
                return a._to_str().startswith(b._to_str())
            return impl