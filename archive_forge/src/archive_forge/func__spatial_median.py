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
def _spatial_median(X, max_iter=300, tol=0.001):
    """Spatial median (L1 median).

    The spatial median is member of a class of so-called M-estimators which
    are defined by an optimization problem. Given a number of p points in an
    n-dimensional space, the point x minimizing the sum of all distances to the
    p other points is called spatial median.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    max_iter : int, default=300
        Maximum number of iterations.

    tol : float, default=1.e-3
        Stop the algorithm if spatial_median has converged.

    Returns
    -------
    spatial_median : ndarray of shape = (n_features,)
        Spatial median.

    n_iter : int
        Number of iterations needed.

    References
    ----------
    - On Computation of Spatial Median for Robust Data Mining, 2005
      T. Kärkkäinen and S. Äyrämö
      http://users.jyu.fi/~samiayr/pdf/ayramo_eurogen05.pdf
    """
    if X.shape[1] == 1:
        return (1, np.median(X.ravel(), keepdims=True))
    tol **= 2
    spatial_median_old = np.mean(X, axis=0)
    for n_iter in range(max_iter):
        spatial_median = _modified_weiszfeld_step(X, spatial_median_old)
        if np.sum((spatial_median_old - spatial_median) ** 2) < tol:
            break
        else:
            spatial_median_old = spatial_median
    else:
        warnings.warn('Maximum number of iterations {max_iter} reached in spatial median for TheilSen regressor.'.format(max_iter=max_iter), ConvergenceWarning)
    return (n_iter, spatial_median)