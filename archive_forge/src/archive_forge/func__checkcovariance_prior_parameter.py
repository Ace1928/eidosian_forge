import math
from numbers import Real
import numpy as np
from scipy.special import betaln, digamma, gammaln
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ._base import BaseMixture, _check_shape
from ._gaussian_mixture import (
def _checkcovariance_prior_parameter(self, X):
    """Check the `covariance_prior_`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
    _, n_features = X.shape
    if self.covariance_prior is None:
        self.covariance_prior_ = {'full': np.atleast_2d(np.cov(X.T)), 'tied': np.atleast_2d(np.cov(X.T)), 'diag': np.var(X, axis=0, ddof=1), 'spherical': np.var(X, axis=0, ddof=1).mean()}[self.covariance_type]
    elif self.covariance_type in ['full', 'tied']:
        self.covariance_prior_ = check_array(self.covariance_prior, dtype=[np.float64, np.float32], ensure_2d=False)
        _check_shape(self.covariance_prior_, (n_features, n_features), '%s covariance_prior' % self.covariance_type)
        _check_precision_matrix(self.covariance_prior_, self.covariance_type)
    elif self.covariance_type == 'diag':
        self.covariance_prior_ = check_array(self.covariance_prior, dtype=[np.float64, np.float32], ensure_2d=False)
        _check_shape(self.covariance_prior_, (n_features,), '%s covariance_prior' % self.covariance_type)
        _check_precision_positivity(self.covariance_prior_, self.covariance_type)
    else:
        self.covariance_prior_ = self.covariance_prior