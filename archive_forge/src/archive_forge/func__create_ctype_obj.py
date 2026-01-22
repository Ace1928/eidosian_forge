import ctypes, ctypes.util, operator, sys
from . import model
@staticmethod
def _create_ctype_obj(init):
    if init is None:
        return ctype()
    return ctype(CTypesPrimitive._to_ctypes(init))