from __future__ import annotations
import operator
from enum import IntEnum
from ._creation_functions import asarray
from ._dtypes import (
from typing import TYPE_CHECKING, Optional, Tuple, Union, Any, SupportsIndex
import types
import cupy as np
from cupy.cuda import Device as _Device
from cupy.cuda import stream as stream_module
from cupy_backends.cuda.api import runtime
from cupy import array_api
@staticmethod
def _normalize_two_args(x1, x2) -> Tuple[Array, Array]:
    """
        Normalize inputs to two arg functions to fix type promotion rules

        NumPy deviates from the spec type promotion rules in cases where one
        argument is 0-dimensional and the other is not. For example:

        >>> import numpy as np
        >>> a = np.array([1.0], dtype=np.float32)
        >>> b = np.array(1.0, dtype=np.float64)
        >>> np.add(a, b) # The spec says this should be float64
        array([2.], dtype=float32)

        To fix this, we add a dimension to the 0-dimension array before passing it
        through. This works because a dimension would be added anyway from
        broadcasting, so the resulting shape is the same, but this prevents NumPy
        from not promoting the dtype.
        """
    if x1.ndim == 0 and x2.ndim != 0:
        x1 = Array._new(x1._array[None])
    elif x2.ndim == 0 and x1.ndim != 0:
        x2 = Array._new(x2._array[None])
    return (x1, x2)