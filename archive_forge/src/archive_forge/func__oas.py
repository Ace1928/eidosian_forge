import warnings
from numbers import Integral, Real
import numpy as np
from ..base import _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, validate_params
from . import EmpiricalCovariance, empirical_covariance
def _oas(X, *, assume_centered=False):
    """Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    The formulation is based on [1]_.
    [1] "Shrinkage algorithms for MMSE covariance estimation.",
        Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
        IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
        https://arxiv.org/pdf/0907.4698.pdf
    """
    if len(X.shape) == 2 and X.shape[1] == 1:
        if not assume_centered:
            X = X - X.mean()
        return (np.atleast_2d((X ** 2).mean()), 0.0)
    n_samples, n_features = X.shape
    emp_cov = empirical_covariance(X, assume_centered=assume_centered)
    alpha = np.mean(emp_cov ** 2)
    mu = np.trace(emp_cov) / n_features
    mu_squared = mu ** 2
    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    shrinkage = 1.0 if den == 0 else min(num / den, 1.0)
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[::n_features + 1] += shrinkage * mu
    return (shrunk_cov, shrinkage)