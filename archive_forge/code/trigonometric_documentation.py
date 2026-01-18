import numpy
import cupy
from cupy import _core
from cupy._math import sumprod
from cupy._math import ufunc
Unwrap by taking the complement of large deltas w.r.t. the period.

    This unwraps a signal `p` by changing elements which have an absolute
    difference from their predecessor of more than ``max(discont, period/2)``
    to their `period`-complementary values.

    For the default case where `period` is :math:`2\pi` and is ``discont``
    is :math:`\pi`, this unwraps a radian phase `p` such that adjacent
    differences are never greater than :math:`\pi` by adding :math:`2k\pi`
    for some integer :math:`k`.

    Args:
        p (cupy.ndarray): Input array.
            discont (float): Maximum discontinuity between values, default is
            ``period/2``. Values below ``period/2`` are treated as if they were
            ``period/2``. To have an effect different from the default,
            ``discont`` should be larger than ``period/2``.
        axis (int): Axis along which unwrap will operate, default is the last
            axis.
        period: float, optional
            Size of the range over which the input wraps. By default, it is
            :math:`2\pi`.
    Returns:
        cupy.ndarray: The result array.

    .. seealso:: :func:`numpy.unwrap`
    