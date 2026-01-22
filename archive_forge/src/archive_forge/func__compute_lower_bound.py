import math
from numbers import Real
import numpy as np
from scipy.special import betaln, digamma, gammaln
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ._base import BaseMixture, _check_shape
from ._gaussian_mixture import (
def _compute_lower_bound(self, log_resp, log_prob_norm):
    """Estimate the lower bound of the model.

        The lower bound on the likelihood (of the training data with respect to
        the model) is used to detect the convergence and has to increase at
        each iteration.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.

        log_prob_norm : float
            Logarithm of the probability of each sample in X.

        Returns
        -------
        lower_bound : float
        """
    n_features, = self.mean_prior_.shape
    log_det_precisions_chol = _compute_log_det_cholesky(self.precisions_cholesky_, self.covariance_type, n_features) - 0.5 * n_features * np.log(self.degrees_of_freedom_)
    if self.covariance_type == 'tied':
        log_wishart = self.n_components * np.float64(_log_wishart_norm(self.degrees_of_freedom_, log_det_precisions_chol, n_features))
    else:
        log_wishart = np.sum(_log_wishart_norm(self.degrees_of_freedom_, log_det_precisions_chol, n_features))
    if self.weight_concentration_prior_type == 'dirichlet_process':
        log_norm_weight = -np.sum(betaln(self.weight_concentration_[0], self.weight_concentration_[1]))
    else:
        log_norm_weight = _log_dirichlet_norm(self.weight_concentration_)
    return -np.sum(np.exp(log_resp) * log_resp) - log_wishart - log_norm_weight - 0.5 * n_features * np.sum(np.log(self.mean_precision_))