import ctypes
import sys
from numba.core import types
from numba.core.typing import templates
from .typeof import typeof_impl
def _convert_internal(ty):
    if isinstance(ty, types.CPointer):
        return ctypes.POINTER(_convert_internal(ty.dtype))
    else:
        return _TO_CTYPES.get(ty)