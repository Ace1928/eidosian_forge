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
def window_transform(self, window):
    """Get the affine transform for a dataset window.

        Parameters
        ----------
        window: rasterio.windows.Window
            Dataset window

        Returns
        -------
        transform: Affine
            The affine transform matrix for the given window

        """
    gtransform = guard_transform(self.transform)
    return transform(window, gtransform)