import ctypes, ctypes.util, operator, sys
from . import model
@classmethod
def _newp(cls, init):
    return cls(init)