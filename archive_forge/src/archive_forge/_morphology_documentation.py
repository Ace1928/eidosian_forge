import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters

    Multidimensional black tophat filter.

    Args:
        input (cupy.ndarray): The input array.
        size (tuple of ints): Shape of a flat and full structuring element used
            for the black tophat. Optional if ``footprint`` or ``structure`` is
            provided.
        footprint (array of ints): Positions of non-infinite elements of a flat
            structuring element used for the black tophat. Non-zero values
            give the set of neighbors of the center over which opening is
            chosen.
        structure (array of ints): Structuring element used for the black
            tophat. ``structure`` may be a non-flat structuring element.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarry : Result of the filter of ``input`` with ``structure``.

    .. seealso:: :func:`scipy.ndimage.black_tophat`
    