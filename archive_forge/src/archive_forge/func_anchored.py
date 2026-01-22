import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
def anchored(self, c, container=None):
    """
        Return a copy of the `Bbox` anchored to *c* within *container*.

        Parameters
        ----------
        c : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}
            Either an (*x*, *y*) pair of relative coordinates (0 is left or
            bottom, 1 is right or top), 'C' (center), or a cardinal direction
            ('SW', southwest, is bottom left, etc.).
        container : `Bbox`, optional
            The box within which the `Bbox` is positioned.

        See Also
        --------
        .Axes.set_anchor
        """
    if container is None:
        _api.warn_deprecated('3.8', message='Calling anchored() with no container bbox returns a frozen copy of the original bbox and is deprecated since %(since)s.')
        container = self
    l, b, w, h = container.bounds
    L, B, W, H = self.bounds
    cx, cy = self.coefs[c] if isinstance(c, str) else c
    return Bbox(self._points + [l + cx * (w - W) - L, b + cy * (h - H) - B])