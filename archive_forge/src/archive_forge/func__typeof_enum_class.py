from collections import namedtuple
from functools import singledispatch
import ctypes
import enum
import numpy as np
from numpy.random.bit_generator import BitGenerator
from numba.core import types, utils, errors
from numba.np import numpy_support
@typeof_impl.register(enum.EnumMeta)
def _typeof_enum_class(val, c):
    cls = val
    members = list(cls.__members__.values())
    if len(members) == 0:
        raise ValueError('Cannot type enum with no members')
    dtypes = {typeof_impl(mem.value, c) for mem in members}
    if len(dtypes) > 1:
        raise ValueError('Cannot type heterogeneous enum: got value types %s' % ', '.join(sorted((str(ty) for ty in dtypes))))
    if issubclass(val, enum.IntEnum):
        typecls = types.IntEnumClass
    else:
        typecls = types.EnumClass
    return typecls(cls, dtypes.pop())