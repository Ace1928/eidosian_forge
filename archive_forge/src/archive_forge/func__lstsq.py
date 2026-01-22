import warnings
from itertools import combinations
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import binom
from ..base import RegressorMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..utils import check_random_state
from ..utils._param_validation import Interval
from ..utils.parallel import Parallel, delayed
from ._base import LinearModel
def _lstsq(X, y, indices, fit_intercept):
    """Least Squares Estimator for TheilSenRegressor class.

    This function calculates the least squares method on a subset of rows of X
    and y defined by the indices array. Optionally, an intercept column is
    added if intercept is set to true.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Design matrix, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : ndarray of shape (n_samples,)
        Target vector, where `n_samples` is the number of samples.

    indices : ndarray of shape (n_subpopulation, n_subsamples)
        Indices of all subsamples with respect to the chosen subpopulation.

    fit_intercept : bool
        Fit intercept or not.

    Returns
    -------
    weights : ndarray of shape (n_subpopulation, n_features + intercept)
        Solution matrix of n_subpopulation solved least square problems.
    """
    fit_intercept = int(fit_intercept)
    n_features = X.shape[1] + fit_intercept
    n_subsamples = indices.shape[1]
    weights = np.empty((indices.shape[0], n_features))
    X_subpopulation = np.ones((n_subsamples, n_features))
    y_subpopulation = np.zeros(max(n_subsamples, n_features))
    lstsq, = get_lapack_funcs(('gelss',), (X_subpopulation, y_subpopulation))
    for index, subset in enumerate(indices):
        X_subpopulation[:, fit_intercept:] = X[subset, :]
        y_subpopulation[:n_subsamples] = y[subset]
        weights[index] = lstsq(X_subpopulation, y_subpopulation)[1][:n_features]
    return weights