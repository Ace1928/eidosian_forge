import math
from numbers import Real
import numpy as np
from scipy.special import betaln, digamma, gammaln
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ._base import BaseMixture, _check_shape
from ._gaussian_mixture import (
def _estimate_precisions(self, nk, xk, sk):
    """Estimate the precisions parameters of the precision distribution.

        Parameters
        ----------
        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)

        sk : array-like
            The shape depends of `covariance_type`:
            'full' : (n_components, n_features, n_features)
            'tied' : (n_features, n_features)
            'diag' : (n_components, n_features)
            'spherical' : (n_components,)
        """
    {'full': self._estimate_wishart_full, 'tied': self._estimate_wishart_tied, 'diag': self._estimate_wishart_diag, 'spherical': self._estimate_wishart_spherical}[self.covariance_type](nk, xk, sk)
    self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, self.covariance_type)