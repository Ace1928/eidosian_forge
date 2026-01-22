import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
Compute a 1D filter along the given axis using the provided raw kernel.

    Unlike the scipy.ndimage function, this does not support the
    ``extra_arguments`` or ``extra_keywordsdict`` arguments and has significant
    restrictions on the ``function`` provided.

    Args:
        input (cupy.ndarray): The input array.
        function (cupy.RawKernel): The kernel to apply along each axis.
        filter_size (int): Length of the filter.
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

    .. note::
        The provided function (as a RawKernel) must have the following
        signature. Unlike most functions, this should not utilize
        `blockDim`/`blockIdx`/`threadIdx`::

            __global__ void func(double *input_line, ptrdiff_t input_length,
                                 double *output_line, ptrdiff_t output_length)

    .. seealso:: :func:`scipy.ndimage.generic_filter1d`
    