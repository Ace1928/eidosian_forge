import warnings
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.stats import chi2
from ..base import _fit_context
from ..utils import check_array, check_random_state
from ..utils._param_validation import Interval
from ..utils.extmath import fast_logdet
from ._empirical_covariance import EmpiricalCovariance, empirical_covariance
def _c_step(X, n_support, random_state, remaining_iterations=30, initial_estimates=None, verbose=False, cov_computation_method=empirical_covariance):
    n_samples, n_features = X.shape
    dist = np.inf
    support = np.zeros(n_samples, dtype=bool)
    if initial_estimates is None:
        support[random_state.permutation(n_samples)[:n_support]] = True
    else:
        location = initial_estimates[0]
        covariance = initial_estimates[1]
        precision = linalg.pinvh(covariance)
        X_centered = X - location
        dist = (np.dot(X_centered, precision) * X_centered).sum(1)
        support[np.argsort(dist)[:n_support]] = True
    X_support = X[support]
    location = X_support.mean(0)
    covariance = cov_computation_method(X_support)
    det = fast_logdet(covariance)
    if np.isinf(det):
        precision = linalg.pinvh(covariance)
    previous_det = np.inf
    while det < previous_det and remaining_iterations > 0 and (not np.isinf(det)):
        previous_location = location
        previous_covariance = covariance
        previous_det = det
        previous_support = support
        precision = linalg.pinvh(covariance)
        X_centered = X - location
        dist = (np.dot(X_centered, precision) * X_centered).sum(axis=1)
        support = np.zeros(n_samples, dtype=bool)
        support[np.argsort(dist)[:n_support]] = True
        X_support = X[support]
        location = X_support.mean(axis=0)
        covariance = cov_computation_method(X_support)
        det = fast_logdet(covariance)
        remaining_iterations -= 1
    previous_dist = dist
    dist = (np.dot(X - location, precision) * (X - location)).sum(axis=1)
    if np.isinf(det):
        results = (location, covariance, det, support, dist)
    if np.allclose(det, previous_det):
        if verbose:
            print('Optimal couple (location, covariance) found before ending iterations (%d left)' % remaining_iterations)
        results = (location, covariance, det, support, dist)
    elif det > previous_det:
        warnings.warn('Determinant has increased; this should not happen: log(det) > log(previous_det) (%.15f > %.15f). You may want to try with a higher value of support_fraction (current value: %.3f).' % (det, previous_det, n_support / n_samples), RuntimeWarning)
        results = (previous_location, previous_covariance, previous_det, previous_support, previous_dist)
    if remaining_iterations == 0:
        if verbose:
            print('Maximum number of iterations reached')
        results = (location, covariance, det, support, dist)
    return results