import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
def uniform_filter1d(input, size, axis=-1, output=None, mode='reflect', cval=0.0, origin=0):
    """One-dimensional uniform filter along the given axis.

    The lines of the array along the given axis are filtered with a uniform
    filter of the given size.

    Args:
        input (cupy.ndarray): The input array.
        size (int): Length of the uniform filter.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.uniform_filter1d`

    .. note::
        When the output data type is integral (or when no output is provided
        and input is integral) the results may not perfectly match the results
        from SciPy due to floating-point rounding of intermediate results.
    """
    weights_dtype = _util._init_weights_dtype(input)
    weights = cupy.full(size, 1 / size, dtype=weights_dtype)
    return correlate1d(input, weights, axis, output, mode, cval, origin)