import numpy as np
from . import _spath
Find the shortest path through an n-d array from one side to another.

    Parameters
    ----------
    arr : ndarray of float64
    reach : int, optional
        By default (``reach = 1``), the shortest path can only move
        one row up or down for every step it moves forward (i.e.,
        the path gradient is limited to 1). `reach` defines the
        number of elements that can be skipped along each non-axis
        dimension at each step.
    axis : int, optional
        The axis along which the path must always move forward (default -1)
    output_indexlist : bool, optional
        See return value `p` for explanation.

    Returns
    -------
    p : iterable of int
        For each step along `axis`, the coordinate of the shortest path.
        If `output_indexlist` is True, then the path is returned as a list of
        n-d tuples that index into `arr`. If False, then the path is returned
        as an array listing the coordinates of the path along the non-axis
        dimensions for each step along the axis dimension. That is,
        `p.shape == (arr.shape[axis], arr.ndim-1)` except that p is squeezed
        before returning so if `arr.ndim == 2`, then
        `p.shape == (arr.shape[axis],)`
    cost : float
        Cost of path.  This is the absolute sum of all the
        differences along the path.

    