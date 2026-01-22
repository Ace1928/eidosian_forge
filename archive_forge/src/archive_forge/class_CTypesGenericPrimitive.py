import ctypes, ctypes.util, operator, sys
from . import model
class CTypesGenericPrimitive(CTypesData):
    __slots__ = []

    def __hash__(self):
        return hash(self._value)

    def _get_own_repr(self):
        return repr(self._from_ctypes(self._value))