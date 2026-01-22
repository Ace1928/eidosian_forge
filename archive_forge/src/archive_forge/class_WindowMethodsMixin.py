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
class WindowMethodsMixin:
    """Mixin providing methods for window-related calculations.
    These methods are wrappers for the functionality in
    `rasterio.windows` module.

    A subclass with this mixin MUST provide the following
    properties: `transform`, `height` and `width`.

    """

    def window(self, left, bottom, right, top, precision=None):
        """Get the window corresponding to the bounding coordinates.

        The resulting window is not cropped to the row and column
        limits of the dataset.

        Parameters
        ----------
        left: float
            Left (west) bounding coordinate
        bottom: float
            Bottom (south) bounding coordinate
        right: float
            Right (east) bounding coordinate
        top: float
            Top (north) bounding coordinate
        precision: int, optional
            This parameter is unused, deprecated in rasterio 1.3.0, and
            will be removed in version 2.0.0.

        Returns
        -------
        window: Window

        """
        if precision is not None:
            warnings.warn('The precision parameter is unused, deprecated, and will be removed in 2.0.0.', RasterioDeprecationWarning)
        return from_bounds(left, bottom, right, top, transform=guard_transform(self.transform))

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