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
def _get_analyze_compat_dtype(arr):
    """Return an Analyze-compatible dtype that ``arr`` can be safely cast to

    Analyze-compatible types are returned without inspection:

    >>> _get_analyze_compat_dtype(np.uint8([0, 1]))
    dtype('uint8')
    >>> _get_analyze_compat_dtype(np.int16([0, 1]))
    dtype('int16')
    >>> _get_analyze_compat_dtype(np.int32([0, 1]))
    dtype('int32')
    >>> _get_analyze_compat_dtype(np.float32([0, 1]))
    dtype('float32')

    Signed ``int8`` are cast to ``uint8`` or ``int16`` based on value ranges:

    >>> _get_analyze_compat_dtype(np.int8([0, 1]))
    dtype('uint8')
    >>> _get_analyze_compat_dtype(np.int8([-1, 1]))
    dtype('int16')

    Unsigned ``uint16`` are cast to ``int16`` or ``int32`` based on value ranges:

    >>> _get_analyze_compat_dtype(np.uint16([32767]))
    dtype('int16')
    >>> _get_analyze_compat_dtype(np.uint16([65535]))
    dtype('int32')

    ``int32`` is returned for integer types and ``float32`` for floating point types:

    >>> _get_analyze_compat_dtype(np.array([-1, 1]))
    dtype('int32')
    >>> _get_analyze_compat_dtype(np.array([-1., 1.]))
    dtype('float32')

    If the value ranges exceed 4 bytes or cannot be cast, then a ``ValueError`` is raised:

    >>> _get_analyze_compat_dtype(np.array([0, 4294967295]))
    Traceback (most recent call last):
       ...
    ValueError: Cannot find analyze-compatible dtype for array with dtype=int64
        (min=0, max=4294967295)

    >>> _get_analyze_compat_dtype([0., 2.e40])
    Traceback (most recent call last):
       ...
    ValueError: Cannot find analyze-compatible dtype for array with dtype=float64
        (min=0.0, max=2e+40)

    Note that real-valued complex arrays cannot be safely cast.

    >>> _get_analyze_compat_dtype(np.array([1+0j]))
    Traceback (most recent call last):
       ...
    ValueError: Cannot find analyze-compatible dtype for array with dtype=complex128
        (min=(1+0j), max=(1+0j))
    """
    arr = np.asanyarray(arr)
    dtype = arr.dtype
    if dtype in (np.uint8, np.int16, np.int32, np.float32):
        return dtype
    if dtype == np.int8:
        return np.dtype('uint8' if arr.min() >= 0 else 'int16')
    elif dtype == np.uint16:
        return np.dtype('int16' if arr.max() <= np.iinfo(np.int16).max else 'int32')
    mn, mx = (arr.min(), arr.max())
    if arr.dtype.kind in 'iu':
        info = np.iinfo('int32')
        if mn >= info.min and mx <= info.max:
            return np.dtype('int32')
    elif arr.dtype.kind == 'f':
        info = np.finfo('float32')
        if mn >= info.min and mx <= info.max:
            return np.dtype('float32')
    raise ValueError(f'Cannot find analyze-compatible dtype for array with dtype={dtype} (min={mn}, max={mx})')