import warnings
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import as_float_array, check_array, check_random_state
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.validation import check_is_fitted
def _gs_decorrelation(w, W, j):
    """
    Orthonormalize w wrt the first j rows of W.

    Parameters
    ----------
    w : ndarray of shape (n,)
        Array to be orthogonalized

    W : ndarray of shape (p, n)
        Null space definition

    j : int < p
        The no of (from the first) rows of Null space W wrt which w is
        orthogonalized.

    Notes
    -----
    Assumes that W is orthogonal
    w changed in place
    """
    w -= np.linalg.multi_dot([w, W[:j].T, W[:j]])
    return w