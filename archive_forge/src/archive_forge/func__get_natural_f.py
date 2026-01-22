import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def _get_natural_f(knots):
    """Returns mapping of natural cubic spline values to 2nd derivatives.

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, pp 145-146

    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: A 2-d array mapping natural cubic spline values at
     knots to second derivatives.

    :raise ImportError: if scipy is not found, required for
     ``linalg.solve_banded()``
    """
    try:
        from scipy import linalg
    except ImportError:
        raise ImportError('Cubic spline functionality requires scipy.')
    h = knots[1:] - knots[:-1]
    diag = (h[:-1] + h[1:]) / 3.0
    ul_diag = h[1:-1] / 6.0
    banded_b = np.array([np.r_[0.0, ul_diag], diag, np.r_[ul_diag, 0.0]])
    d = np.zeros((knots.size - 2, knots.size))
    for i in range(knots.size - 2):
        d[i, i] = 1.0 / h[i]
        d[i, i + 2] = 1.0 / h[i + 1]
        d[i, i + 1] = -d[i, i] - d[i, i + 2]
    fm = linalg.solve_banded((1, 1), banded_b, d)
    return np.vstack([np.zeros(knots.size), fm, np.zeros(knots.size)])