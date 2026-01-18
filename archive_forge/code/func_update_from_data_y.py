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
def update_from_data_y(self, y, ignore=None):
    """
        Update the y-bounds of the `Bbox` based on the passed in data. After
        updating, the bounds will have positive *height*, and *y0* will be the
        minimal value.

        Parameters
        ----------
        y : `~numpy.ndarray`
            Array of y-values.
        ignore : bool, optional
            - When ``True``, ignore the existing bounds of the `Bbox`.
            - When ``False``, include the existing bounds of the `Bbox`.
            - When ``None``, use the last value passed to :meth:`ignore`.
        """
    y = np.ravel(y)
    self.update_from_data_xy(np.column_stack([np.ones(y.size), y]), ignore=ignore, updatex=False)