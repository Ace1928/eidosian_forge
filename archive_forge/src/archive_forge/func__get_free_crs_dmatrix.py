import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def _get_free_crs_dmatrix(x, knots, cyclic=False):
    """Builds an unconstrained cubic regression spline design matrix.

    Returns design matrix with dimensions ``len(x) x n``
    for a cubic regression spline smoother
    where
     - ``n = len(knots)`` for natural CRS
     - ``n = len(knots) - 1`` for cyclic CRS

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, p. 145

    :param x: The 1-d array values.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :param cyclic: Indicates whether used cubic regression splines should
     be cyclic or not. Default is ``False``.
    :return: The (2-d array) design matrix.
    """
    n = knots.size
    if cyclic:
        x = _map_cyclic(x, min(knots), max(knots))
        n -= 1
    ajm, ajp, cjm, cjp, j = _compute_base_functions(x, knots)
    j1 = j + 1
    if cyclic:
        j1[j1 == n] = 0
    i = np.identity(n)
    if cyclic:
        f = _get_cyclic_f(knots)
    else:
        f = _get_natural_f(knots)
    dmt = ajm * i[j, :].T + ajp * i[j1, :].T + cjm * f[j, :].T + cjp * f[j1, :].T
    return dmt.T