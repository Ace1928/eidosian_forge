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
def update_from_path(self, path, ignore=None, updatex=True, updatey=True):
    """
        Update the bounds of the `Bbox` to contain the vertices of the
        provided path. After updating, the bounds will have positive *width*
        and *height*; *x0* and *y0* will be the minimal values.

        Parameters
        ----------
        path : `~matplotlib.path.Path`
        ignore : bool, optional
           - When ``True``, ignore the existing bounds of the `Bbox`.
           - When ``False``, include the existing bounds of the `Bbox`.
           - When ``None``, use the last value passed to :meth:`ignore`.
        updatex, updatey : bool, default: True
            When ``True``, update the x/y values.
        """
    if ignore is None:
        ignore = self._ignore
    if path.vertices.size == 0:
        return
    points, minpos, changed = update_path_extents(path, None, self._points, self._minpos, ignore)
    if changed:
        self.invalidate()
        if updatex:
            self._points[:, 0] = points[:, 0]
            self._minpos[0] = minpos[0]
        if updatey:
            self._points[:, 1] = points[:, 1]
            self._minpos[1] = minpos[1]