import ctypes, ctypes.util, operator, sys
from . import model
@staticmethod
def _to_ctypes(x):
    if not isinstance(x, (int, long, float, CTypesData)):
        raise TypeError('float expected, got %s' % type(x).__name__)
    return ctype(x).value