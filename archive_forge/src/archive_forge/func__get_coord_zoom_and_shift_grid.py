import numpy
import cupy
import cupy._core.internal
from cupyx.scipy.ndimage import _spline_prefilter_core
from cupyx.scipy.ndimage import _spline_kernel_weights
from cupyx.scipy.ndimage import _util
def _get_coord_zoom_and_shift_grid(ndim, nprepad=0):
    """Compute target coordinate based on a shift followed by a zoom.

    This version zooms from the outer edges of the grid pixels.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        zoom[ndim]: array containing the zoom for each axis
        shift[ndim]: array containing the zoom for each axis

    computes::

        c_j = zoom[j] * (in_coord[j] - shift[j] + 0.5) - 0.5

    """
    ops = []
    pre = f' + (W){nprepad}' if nprepad > 0 else ''
    for j in range(ndim):
        ops.append(f'\n    W c_{j} = zoom[{j}] * ((W)in_coord[{j}] - shift[j] + 0.5) - 0.5{pre};')
    return ops