import sys
import warnings
from math import log
from numbers import Integral, Real
import numpy as np
from scipy import interpolate, linalg
from scipy.linalg.lapack import get_lapack_funcs
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..model_selection import check_cv
from ..utils import (  # type: ignore
from ..utils._metadata_requests import (
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.parallel import Parallel, delayed
from ._base import LinearModel, LinearRegression, _preprocess_data
def _lars_path_residues(X_train, y_train, X_test, y_test, Gram=None, copy=True, method='lar', verbose=False, fit_intercept=True, max_iter=500, eps=np.finfo(float).eps, positive=False):
    """Compute the residues on left-out data for a full LARS path

    Parameters
    -----------
    X_train : array-like of shape (n_samples, n_features)
        The data to fit the LARS on

    y_train : array-like of shape (n_samples,)
        The target variable to fit LARS on

    X_test : array-like of shape (n_samples, n_features)
        The data to compute the residues on

    y_test : array-like of shape (n_samples,)
        The target variable to compute the residues on

    Gram : None, 'auto' or array-like of shape (n_features, n_features),             default=None
        Precomputed Gram matrix (X' * X), if ``'auto'``, the Gram
        matrix is precomputed from the given X, if there are more samples
        than features

    copy : bool, default=True
        Whether X_train, X_test, y_train and y_test should be copied;
        if False, they may be overwritten.

    method : {'lar' , 'lasso'}, default='lar'
        Specifies the returned model. Select ``'lar'`` for Least Angle
        Regression, ``'lasso'`` for the Lasso.

    verbose : bool or int, default=False
        Sets the amount of verbosity

    fit_intercept : bool, default=True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    positive : bool, default=False
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.
        See reservations for using this option in combination with method
        'lasso' for expected small values of alpha in the doc of LassoLarsCV
        and LassoLarsIC.

    max_iter : int, default=500
        Maximum number of iterations to perform.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    Returns
    --------
    alphas : array-like of shape (n_alphas,)
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter`` or ``n_features``, whichever
        is smaller.

    active : list
        Indices of active variables at the end of the path.

    coefs : array-like of shape (n_features, n_alphas)
        Coefficients along the path

    residues : array-like of shape (n_alphas, n_samples)
        Residues of the prediction on the test data
    """
    X_train = _check_copy_and_writeable(X_train, copy)
    y_train = _check_copy_and_writeable(y_train, copy)
    X_test = _check_copy_and_writeable(X_test, copy)
    y_test = _check_copy_and_writeable(y_test, copy)
    if fit_intercept:
        X_mean = X_train.mean(axis=0)
        X_train -= X_mean
        X_test -= X_mean
        y_mean = y_train.mean(axis=0)
        y_train = as_float_array(y_train, copy=False)
        y_train -= y_mean
        y_test = as_float_array(y_test, copy=False)
        y_test -= y_mean
    alphas, active, coefs = lars_path(X_train, y_train, Gram=Gram, copy_X=False, copy_Gram=False, method=method, verbose=max(0, verbose - 1), max_iter=max_iter, eps=eps, positive=positive)
    residues = np.dot(X_test, coefs) - y_test[:, np.newaxis]
    return (alphas, active, coefs, residues.T)