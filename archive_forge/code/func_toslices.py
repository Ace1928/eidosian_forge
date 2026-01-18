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
def toslices(self):
    """Slice objects for use as an ndarray indexer.

        Returns
        -------
        row_slice, col_slice: slice
            A pair of slices in row, column order

        """
    (r0, r1), (c0, c1) = self.toranges()
    if r0 < 0:
        r0 = 0
    if r1 < 0:
        r1 = 0
    if c0 < 0:
        c0 = 0
    if c1 < 0:
        c1 = 0
    return (slice(int(math.floor(r0)), int(math.ceil(r1))), slice(int(math.floor(c0)), int(math.ceil(c1))))