import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.linalg import svd
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_consistent_length
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import svd_flip
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import FLOAT_DTYPES, check_is_fitted
def _get_first_singular_vectors_power_method(X, Y, mode='A', max_iter=500, tol=1e-06, norm_y_weights=False):
    """Return the first left and right singular vectors of X'Y.

    Provides an alternative to the svd(X'Y) and uses the power method instead.
    With norm_y_weights to True and in mode A, this corresponds to the
    algorithm section 11.3 of the Wegelin's review, except this starts at the
    "update saliences" part.
    """
    eps = np.finfo(X.dtype).eps
    try:
        y_score = next((col for col in Y.T if np.any(np.abs(col) > eps)))
    except StopIteration as e:
        raise StopIteration('Y residual is constant') from e
    x_weights_old = 100
    if mode == 'B':
        X_pinv, Y_pinv = (_pinv2_old(X), _pinv2_old(Y))
    for i in range(max_iter):
        if mode == 'B':
            x_weights = np.dot(X_pinv, y_score)
        else:
            x_weights = np.dot(X.T, y_score) / np.dot(y_score, y_score)
        x_weights /= np.sqrt(np.dot(x_weights, x_weights)) + eps
        x_score = np.dot(X, x_weights)
        if mode == 'B':
            y_weights = np.dot(Y_pinv, x_score)
        else:
            y_weights = np.dot(Y.T, x_score) / np.dot(x_score.T, x_score)
        if norm_y_weights:
            y_weights /= np.sqrt(np.dot(y_weights, y_weights)) + eps
        y_score = np.dot(Y, y_weights) / (np.dot(y_weights, y_weights) + eps)
        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        x_weights_old = x_weights
    n_iter = i + 1
    if n_iter == max_iter:
        warnings.warn('Maximum number of iterations reached', ConvergenceWarning)
    return (x_weights, y_weights, n_iter)