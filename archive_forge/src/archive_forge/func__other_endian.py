import sys
from ctypes import *
def _other_endian(typ):
    """Return the type with the 'other' byte order.  Simple types like
    c_int and so on already have __ctype_be__ and __ctype_le__
    attributes which contain the types, for more complicated types
    arrays and structures are supported.
    """
    if hasattr(typ, _OTHER_ENDIAN):
        return getattr(typ, _OTHER_ENDIAN)
    if isinstance(typ, _array_type):
        return _other_endian(typ._type_) * typ._length_
    if issubclass(typ, (Structure, Union)):
        return typ
    raise TypeError('This type does not support other endian: %s' % typ)