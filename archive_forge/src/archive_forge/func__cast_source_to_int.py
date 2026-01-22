import ctypes, ctypes.util, operator, sys
from . import model
def _cast_source_to_int(source):
    if isinstance(source, (int, long, float)):
        source = int(source)
    elif isinstance(source, CTypesData):
        source = source._cast_to_integer()
    elif isinstance(source, bytes):
        source = ord(source)
    elif source is None:
        source = 0
    else:
        raise TypeError('bad type for cast to %r: %r' % (CTypesPrimitive, type(source).__name__))
    return source