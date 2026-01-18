import ctypes
import sys
from numba.core import types
from numba.core.typing import templates
from .typeof import typeof_impl
def to_ctypes(ty):
    """
    Convert the given Numba type to a ctypes type.
    """
    assert isinstance(ty, types.Type), ty
    if ty is types.none:
        return None

    def _convert_internal(ty):
        if isinstance(ty, types.CPointer):
            return ctypes.POINTER(_convert_internal(ty.dtype))
        else:
            return _TO_CTYPES.get(ty)
    ctypeobj = _convert_internal(ty)
    if ctypeobj is None:
        raise TypeError("Cannot convert Numba type '%s' to ctypes type" % (ty,))
    return ctypeobj