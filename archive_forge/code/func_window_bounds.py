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
def window_bounds(self, window):
    """Get the bounds of a window

        Parameters
        ----------
        window: rasterio.windows.Window
            Dataset window

        Returns
        -------
        bounds : tuple
            x_min, y_min, x_max, y_max for the given window

        """
    transform = guard_transform(self.transform)
    return bounds(window, transform)