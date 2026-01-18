import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@register_jitable(_nrt=False)
def unicode_charseq_get_code(a, i):
    """Access i-th item of UnicodeCharSeq object via code value
    """
    if unicode_byte_width == 4:
        return deref_uint32(a, i)
    elif unicode_byte_width == 2:
        return deref_uint16(a, i)
    elif unicode_byte_width == 1:
        return deref_uint8(a, i)
    else:
        raise NotImplementedError('unicode_charseq_get_code: unicode_byte_width not in [1, 2, 4]')