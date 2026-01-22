import ctypes, ctypes.util, operator, sys
from . import model
@classmethod
def _offsetof(cls, fieldname):
    return getattr(cls._ctype, fieldname).offset