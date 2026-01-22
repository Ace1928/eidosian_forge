import ctypes, ctypes.util, operator, sys
from . import model
@classmethod
def _alignment(cls):
    return ctypes.alignment(cls._ctype)