import ctypes
from ...base import _LIB, check_call
from . import function
from .types import RETURN_SWITCH, TypeCode
class ObjectBase(object):
    """Base object for all object types"""
    __slots__ = ['handle']

    def __del__(self):
        if _LIB is not None:
            check_call(_LIB.MXNetObjectFree(self.handle))