import os
from numpy import (
from numpy.core.multiarray import _flagdict, flagsobj
def as_ctypes(obj):
    """Create and return a ctypes object from a numpy array.  Actually
        anything that exposes the __array_interface__ is accepted."""
    ai = obj.__array_interface__
    if ai['strides']:
        raise TypeError('strided arrays not supported')
    if ai['version'] != 3:
        raise TypeError('only __array_interface__ version 3 supported')
    addr, readonly = ai['data']
    if readonly:
        raise TypeError('readonly arrays unsupported')
    ctype_scalar = as_ctypes_type(ai['typestr'])
    result_type = _ctype_ndarray(ctype_scalar, ai['shape'])
    result = result_type.from_address(addr)
    result.__keep = obj
    return result