import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def _find_knots_lower_bounds(x, knots):
    """Finds knots lower bounds for given values.

    Returns an array of indices ``I`` such that
    ``0 <= I[i] <= knots.size - 2`` for all ``i``
    and
    ``knots[I[i]] < x[i] <= knots[I[i] + 1]`` if
    ``np.min(knots) < x[i] <= np.max(knots)``,
    ``I[i] = 0`` if ``x[i] <= np.min(knots)``
    ``I[i] = knots.size - 2`` if ``np.max(knots) < x[i]``

    :param x: The 1-d array values whose knots lower bounds are to be found.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: An array of knots lower bounds indices.
    """
    lb = np.searchsorted(knots, x) - 1
    lb[lb == -1] = 0
    lb[lb == knots.size - 1] = knots.size - 2
    return lb