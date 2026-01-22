import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def _safely_castable_to_int(dt):
    """Test whether the NumPy data type `dt` can be safely cast to an int."""
    int_size = cupy.dtype(int).itemsize
    safe = cupy.issubdtype(dt, cupy.signedinteger) and dt.itemsize <= int_size or (cupy.issubdtype(dt, cupy.unsignedinteger) and dt.itemsize < int_size)
    return safe