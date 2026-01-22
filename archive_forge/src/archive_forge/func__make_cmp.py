import ctypes, ctypes.util, operator, sys
from . import model
def _make_cmp(name):
    cmpfunc = getattr(operator, name)

    def cmp(self, other):
        v_is_ptr = not isinstance(self, CTypesGenericPrimitive)
        w_is_ptr = isinstance(other, CTypesData) and (not isinstance(other, CTypesGenericPrimitive))
        if v_is_ptr and w_is_ptr:
            return cmpfunc(self._convert_to_address(None), other._convert_to_address(None))
        elif v_is_ptr or w_is_ptr:
            return NotImplemented
        else:
            if isinstance(self, CTypesGenericPrimitive):
                self = self._value
            if isinstance(other, CTypesGenericPrimitive):
                other = other._value
            return cmpfunc(self, other)
    cmp.func_name = name
    return cmp