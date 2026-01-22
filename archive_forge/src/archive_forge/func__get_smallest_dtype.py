from __future__ import annotations
import warnings
from io import BytesIO
import numpy as np
import numpy.linalg as npl
from . import analyze  # module import
from .arrayproxy import get_obj_dtype
from .batteryrunners import Report
from .casting import have_binary128
from .deprecated import alert_future_error
from .filebasedimages import ImageFileError, SerializableImage
from .optpkg import optional_package
from .quaternions import fillpositive, mat2quat, quat2mat
from .spatialimages import HeaderDataError
from .spm99analyze import SpmAnalyzeHeader
from .volumeutils import Recoder, endian_codes, make_dt_codes
def _get_smallest_dtype(arr, itypes=(np.uint8, np.int16, np.int32), ftypes=()):
    """Return the smallest "sensible" dtype that will hold the array data

    The purpose of this function is to support automatic type selection
    for serialization, so "sensible" here means well-supported in the NIfTI-1 world.

    For floating point data, select between single- and double-precision.
    For integer data, select among uint8, int16 and int32.

    The test is for min/max range, so float64 is pretty unlikely to be hit.

    Returns ``None`` if these dtypes do not suffice.

    >>> _get_smallest_dtype(np.array([0, 1]))
    dtype('uint8')
    >>> _get_smallest_dtype(np.array([-1, 1]))
    dtype('int16')
    >>> _get_smallest_dtype(np.array([0, 256]))
    dtype('int16')
    >>> _get_smallest_dtype(np.array([-65536, 65536]))
    dtype('int32')
    >>> _get_smallest_dtype(np.array([-2147483648, 2147483648]))

    By default floating point types are not searched:

    >>> _get_smallest_dtype(np.array([1.]))
    >>> _get_smallest_dtype(np.array([2. ** 1000]))
    >>> _get_smallest_dtype(np.longdouble(2) ** 2000)
    >>> _get_smallest_dtype(np.array([1+0j]))

    However, this function can be passed "legal" floating point types, and
    the logic works the same.

    >>> _get_smallest_dtype(np.array([1.]), ftypes=('float32',))
    dtype('float32')
    >>> _get_smallest_dtype(np.array([2. ** 1000]), ftypes=('float32',))
    >>> _get_smallest_dtype(np.longdouble(2) ** 2000, ftypes=('float32',))
    >>> _get_smallest_dtype(np.array([1+0j]), ftypes=('float32',))
    """
    arr = np.asanyarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        test_dts = ftypes
        info = np.finfo
    elif np.issubdtype(arr.dtype, np.integer):
        test_dts = itypes
        info = np.iinfo
    else:
        return None
    mn, mx = (np.min(arr), np.max(arr))
    for dt in test_dts:
        dtinfo = info(dt)
        if dtinfo.min <= mn and mx <= dtinfo.max:
            return np.dtype(dt)