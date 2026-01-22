import ctypes, ctypes.util, operator, sys
from . import model
def _convert_to_address(self, BClass):
    if BClass in (CTypesPtr, None) or BClass._automatic_casts:
        return ctypes.addressof(self._blob)
    else:
        return CTypesData._convert_to_address(self, BClass)