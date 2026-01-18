import sys
import operator
import numpy as np
from llvmlite.ir import IntType, Constant
from numba.core.cgutils import is_nonelike
from numba.core.extending import (
from numba.core.imputils import (lower_constant, lower_cast, lower_builtin,
from numba.core.datamodel import register_default, StructModel
from numba.core import types, cgutils
from numba.core.utils import PYVERSION
from numba.core.pythonapi import (
from numba._helperlib import c_helpers
from numba.cpython.hashing import _Py_hash_t
from numba.core.unsafe.bytes import memcpy_region
from numba.core.errors import TypingError
from numba.cpython.unicode_support import (_Py_TOUPPER, _Py_TOLOWER, _Py_UCS4,
from numba.cpython import slicing
@overload_method(types.UnicodeType, 'replace')
def unicode_replace(s, old_str, new_str, count=-1):
    thety = count
    if isinstance(count, types.Omitted):
        thety = count.value
    elif isinstance(count, types.Optional):
        thety = count.type
    if not isinstance(thety, (int, types.Integer)):
        raise TypingError('Unsupported parameters. The parameters must be Integer. Given count: {}'.format(count))
    if not isinstance(old_str, (types.UnicodeType, types.NoneType)):
        raise TypingError('The object must be a UnicodeType. Given: {}'.format(old_str))
    if not isinstance(new_str, types.UnicodeType):
        raise TypingError('The object must be a UnicodeType. Given: {}'.format(new_str))

    def impl(s, old_str, new_str, count=-1):
        if count == 0:
            return s
        if old_str == '':
            schars = list(s)
            if count == -1:
                return new_str + new_str.join(schars) + new_str
            split_result = [new_str]
            min_count = min(len(schars), count)
            for i in range(min_count):
                split_result.append(schars[i])
                if i + 1 != min_count:
                    split_result.append(new_str)
                else:
                    split_result.append(''.join(schars[i + 1:]))
            if count > len(schars):
                split_result.append(new_str)
            return ''.join(split_result)
        schars = s.split(old_str, count)
        result = new_str.join(schars)
        return result
    return impl