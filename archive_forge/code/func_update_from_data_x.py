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
def update_from_data_x(self, x, ignore=None):
    """
        Update the x-bounds of the `Bbox` based on the passed in data. After
        updating, the bounds will have positive *width*, and *x0* will be the
        minimal value.

        Parameters
        ----------
        x : `~numpy.ndarray`
            Array of x-values.
        ignore : bool, optional
           - When ``True``, ignore the existing bounds of the `Bbox`.
           - When ``False``, include the existing bounds of the `Bbox`.
           - When ``None``, use the last value passed to :meth:`ignore`.
        """
    x = np.ravel(x)
    self.update_from_data_xy(np.column_stack([x, np.ones(x.size)]), ignore=ignore, updatey=False)