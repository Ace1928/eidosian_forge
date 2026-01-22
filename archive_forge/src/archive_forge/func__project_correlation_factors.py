import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (
def _project_correlation_factors(X):
    """
    Project a matrix into the domain of matrices whose row-wise sums
    of squares are less than or equal to 1.

    The input matrix is modified in-place.
    """
    nm = np.sqrt((X * X).sum(1))
    ii = np.flatnonzero(nm > 1)
    if len(ii) > 0:
        X[ii, :] /= nm[ii][:, None]