import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def wide_dtype_for(arr):
    arr = np.asarray(arr)
    if safe_issubdtype(arr.dtype, np.integer) or safe_issubdtype(arr.dtype, np.floating):
        return widest_float
    elif safe_issubdtype(arr.dtype, np.complexfloating):
        return widest_complex
    raise ValueError('cannot widen a non-numeric type %r' % (arr.dtype,))