import warnings
import cupy
from cupy import _core
from cupy._core import _routines_statistics as _statistics
from cupy._core import _fusion_thread_local
from cupy._logic import content
Computes the q-th quantile of the data along the specified axis.

    Args:
        a (cupy.ndarray): Array for which to compute quantiles.
        q (float, tuple of floats or cupy.ndarray): Quantiles to compute
            in the range between 0 and 1 inclusive.
        axis (int or tuple of ints): Along which axis or axes to compute the
            quantiles. The flattened array is used by default.
        out (cupy.ndarray): Output array.
        overwrite_input (bool): If True, then allow the input array `a`
            to be modified by the intermediate calculations, to save
            memory. In this case, the contents of the input `a` after this
            function completes is undefined.
        method (str): Interpolation method when a quantile lies between
            two data points. ``linear`` interpolation is used by default.
            Supported interpolations are``lower``, ``higher``, ``midpoint``,
            ``nearest`` and ``linear``.
        keepdims (bool): If ``True``, the axis is remained as an axis of
            size one.
        interpolation (str): Deprecated name for the method keyword argument.

    Returns:
        cupy.ndarray: The quantiles of ``a``, along the axis if specified.

    .. seealso:: :func:`numpy.quantile`
    