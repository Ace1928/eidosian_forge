import functools
import numpy
import cupy
from cupy._core import _routines_statistics as _statistics
Returns the standard deviation along an axis ignoring NaN values.

    Args:
        a (cupy.ndarray): Array to compute standard deviation.
        axis (int): Along which axis to compute standard deviation. The
            flattened array is used by default.
        dtype: Data type specifier.
        out (cupy.ndarray): Output array.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.

    Returns:
        cupy.ndarray: The standard deviation of the input array along the axis.

    .. seealso:: :func:`numpy.nanstd`

    