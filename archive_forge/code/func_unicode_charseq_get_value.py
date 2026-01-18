import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@register_jitable
def unicode_charseq_get_value(a, i):
    """Access i-th item of UnicodeCharSeq object via unicode value

    null code is interpreted as IndexError
    """
    code = unicode_charseq_get_code(a, i)
    if code == 0:
        raise IndexError('index out of range')
    return np.array(code, unicode_uint).view(u1_dtype)[()]