import math
import warnings
import cupy
import numpy
from cupy import _core
from cupy._core import internal
from cupy.cuda import runtime
from cupyx import _texture
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _interp_kernels
from cupyx.scipy.ndimage import _spline_prefilter_core
Zoom an array.

    The array is zoomed using spline interpolation of the requested order.

    Args:
        input (cupy.ndarray): The input array.
        zoom (float or sequence): The zoom factor along the axes. If a float,
            ``zoom`` is the same for each axis. If a sequence, ``zoom`` should
            contain one value for each axis.
        output (cupy.ndarray or ~cupy.dtype): The array in which to place the
            output, or the dtype of the returned array.
        order (int): The order of the spline interpolation, default is 3. Must
            be in the range 0-5.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'reflect'``, ``'wrap'``, ``'grid-mirror'``,
            ``'grid-wrap'``, ``'grid-constant'`` or ``'opencv'``).
        cval (scalar): Value used for points outside the boundaries of
            the input if ``mode='constant'`` or ``mode='opencv'``. Default is
            0.0
        prefilter (bool): Determines if the input array is prefiltered with
            ``spline_filter`` before interpolation. The default is True, which
            will create a temporary ``float64`` array of filtered values if
            ``order > 1``. If setting this to False, the output will be
            slightly blurred if ``order > 1``, unless the input is prefiltered,
            i.e. it is the result of calling ``spline_filter`` on the original
            input.
        grid_mode (bool, optional): If False, the distance from the pixel
            centers is zoomed. Otherwise, the distance including the full pixel
            extent is used. For example, a 1d signal of length 5 is considered
            to have length 4 when ``grid_mode`` is False, but length 5 when
            ``grid_mode`` is True. See the following visual illustration:

            .. code-block:: text

                    | pixel 1 | pixel 2 | pixel 3 | pixel 4 | pixel 5 |
                         |<-------------------------------------->|
                                            vs.
                    |<----------------------------------------------->|

            The starting point of the arrow in the diagram above corresponds to
            coordinate location 0 in each mode.

    Returns:
        cupy.ndarray or None:
            The zoomed input.

    .. seealso:: :func:`scipy.ndimage.zoom`
    