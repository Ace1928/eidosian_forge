from numbers import Integral, Real
import numpy as np
from scipy import optimize
from ..base import BaseEstimator, RegressorMixin, _fit_context
from ..utils import axis0_safe_slice
from ..utils._param_validation import Interval
from ..utils.extmath import safe_sparse_dot
from ..utils.optimize import _check_optimize_result
from ..utils.validation import _check_sample_weight
from ._base import LinearModel
def _huber_loss_and_gradient(w, X, y, epsilon, alpha, sample_weight=None):
    """Returns the Huber loss and the gradient.

    Parameters
    ----------
    w : ndarray, shape (n_features + 1,) or (n_features + 2,)
        Feature vector.
        w[:n_features] gives the coefficients
        w[-1] gives the scale factor and if the intercept is fit w[-2]
        gives the intercept factor.

    X : ndarray of shape (n_samples, n_features)
        Input data.

    y : ndarray of shape (n_samples,)
        Target vector.

    epsilon : float
        Robustness of the Huber estimator.

    alpha : float
        Regularization parameter.

    sample_weight : ndarray of shape (n_samples,), default=None
        Weight assigned to each sample.

    Returns
    -------
    loss : float
        Huber loss.

    gradient : ndarray, shape (len(w))
        Returns the derivative of the Huber loss with respect to each
        coefficient, intercept and the scale as a vector.
    """
    _, n_features = X.shape
    fit_intercept = n_features + 2 == w.shape[0]
    if fit_intercept:
        intercept = w[-2]
    sigma = w[-1]
    w = w[:n_features]
    n_samples = np.sum(sample_weight)
    linear_loss = y - safe_sparse_dot(X, w)
    if fit_intercept:
        linear_loss -= intercept
    abs_linear_loss = np.abs(linear_loss)
    outliers_mask = abs_linear_loss > epsilon * sigma
    outliers = abs_linear_loss[outliers_mask]
    num_outliers = np.count_nonzero(outliers_mask)
    n_non_outliers = X.shape[0] - num_outliers
    outliers_sw = sample_weight[outliers_mask]
    n_sw_outliers = np.sum(outliers_sw)
    outlier_loss = 2.0 * epsilon * np.sum(outliers_sw * outliers) - sigma * n_sw_outliers * epsilon ** 2
    non_outliers = linear_loss[~outliers_mask]
    weighted_non_outliers = sample_weight[~outliers_mask] * non_outliers
    weighted_loss = np.dot(weighted_non_outliers.T, non_outliers)
    squared_loss = weighted_loss / sigma
    if fit_intercept:
        grad = np.zeros(n_features + 2)
    else:
        grad = np.zeros(n_features + 1)
    X_non_outliers = -axis0_safe_slice(X, ~outliers_mask, n_non_outliers)
    grad[:n_features] = 2.0 / sigma * safe_sparse_dot(weighted_non_outliers, X_non_outliers)
    signed_outliers = np.ones_like(outliers)
    signed_outliers_mask = linear_loss[outliers_mask] < 0
    signed_outliers[signed_outliers_mask] = -1.0
    X_outliers = axis0_safe_slice(X, outliers_mask, num_outliers)
    sw_outliers = sample_weight[outliers_mask] * signed_outliers
    grad[:n_features] -= 2.0 * epsilon * safe_sparse_dot(sw_outliers, X_outliers)
    grad[:n_features] += alpha * 2.0 * w
    grad[-1] = n_samples
    grad[-1] -= n_sw_outliers * epsilon ** 2
    grad[-1] -= squared_loss / sigma
    if fit_intercept:
        grad[-2] = -2.0 * np.sum(weighted_non_outliers) / sigma
        grad[-2] -= 2.0 * epsilon * np.sum(sw_outliers)
    loss = n_samples * sigma + squared_loss + outlier_loss
    loss += alpha * np.dot(w, w)
    return (loss, grad)