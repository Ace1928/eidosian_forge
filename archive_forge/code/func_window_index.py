import collections
from collections.abc import Iterable
import functools
import math
import warnings
from affine import Affine
import attr
import numpy as np
from rasterio.errors import WindowError, RasterioDeprecationWarning
from rasterio.transform import rowcol, guard_transform
def window_index(window, height=0, width=0):
    """Construct a pair of slice objects for ndarray indexing

    Starting indexes are rounded down, Stopping indexes are rounded up.

    Parameters
    ----------
    window: Window
        The input window.

    Returns
    -------
    row_slice, col_slice: slice
        A pair of slices in row, column order

    """
    window = evaluate(window, height=height, width=width)
    return window.toslices()