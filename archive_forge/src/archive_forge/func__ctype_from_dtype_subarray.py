import os
from numpy import (
from numpy.core.multiarray import _flagdict, flagsobj
def _ctype_from_dtype_subarray(dtype):
    element_dtype, shape = dtype.subdtype
    ctype = _ctype_from_dtype(element_dtype)
    return _ctype_ndarray(ctype, shape)