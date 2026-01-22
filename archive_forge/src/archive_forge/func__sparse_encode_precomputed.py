import itertools
import sys
import time
from numbers import Integral, Real
from warnings import warn
import numpy as np
from joblib import effective_n_jobs
from scipy import linalg
from ..base import (
from ..linear_model import Lars, Lasso, LassoLars, orthogonal_mp_gram
from ..utils import check_array, check_random_state, gen_batches, gen_even_slices
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.extmath import randomized_svd, row_norms, svd_flip
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
def _sparse_encode_precomputed(X, dictionary, *, gram=None, cov=None, algorithm='lasso_lars', regularization=None, copy_cov=True, init=None, max_iter=1000, verbose=0, positive=False):
    """Generic sparse coding with precomputed Gram and/or covariance matrices.

    Each row of the result is the solution to a Lasso problem.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.

    dictionary : ndarray of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows.

    gram : ndarray of shape (n_components, n_components), default=None
        Precomputed Gram matrix, `dictionary * dictionary'`
        gram can be `None` if method is 'threshold'.

    cov : ndarray of shape (n_components, n_samples), default=None
        Precomputed covariance, `dictionary * X'`.

    algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'},             default='lasso_lars'
        The algorithm used:

        * `'lars'`: uses the least angle regression method
          (`linear_model.lars_path`);
        * `'lasso_lars'`: uses Lars to compute the Lasso solution;
        * `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). lasso_lars will be faster if
          the estimated components are sparse;
        * `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution;
        * `'threshold'`: squashes to zero all coefficients less than
          regularization from the projection `dictionary * data'`.

    regularization : int or float, default=None
        The regularization parameter. It corresponds to alpha when
        algorithm is `'lasso_lars'`, `'lasso_cd'` or `'threshold'`.
        Otherwise it corresponds to `n_nonzero_coefs`.

    init : ndarray of shape (n_samples, n_components), default=None
        Initialization value of the sparse code. Only used if
        `algorithm='lasso_cd'`.

    max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `'lasso_lars'`.

    copy_cov : bool, default=True
        Whether to copy the precomputed covariance matrix; if `False`, it may
        be overwritten.

    verbose : int, default=0
        Controls the verbosity; the higher, the more messages.

    positive: bool, default=False
        Whether to enforce a positivity constraint on the sparse code.

        .. versionadded:: 0.20

    Returns
    -------
    code : ndarray of shape (n_components, n_features)
        The sparse codes.
    """
    n_samples, n_features = X.shape
    n_components = dictionary.shape[0]
    if algorithm == 'lasso_lars':
        alpha = float(regularization) / n_features
        try:
            err_mgt = np.seterr(all='ignore')
            lasso_lars = LassoLars(alpha=alpha, fit_intercept=False, verbose=verbose, precompute=gram, fit_path=False, positive=positive, max_iter=max_iter)
            lasso_lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lasso_lars.coef_
        finally:
            np.seterr(**err_mgt)
    elif algorithm == 'lasso_cd':
        alpha = float(regularization) / n_features
        clf = Lasso(alpha=alpha, fit_intercept=False, precompute=gram, max_iter=max_iter, warm_start=True, positive=positive)
        if init is not None:
            if not init.flags['WRITEABLE']:
                init = np.array(init)
            clf.coef_ = init
        clf.fit(dictionary.T, X.T, check_input=False)
        new_code = clf.coef_
    elif algorithm == 'lars':
        try:
            err_mgt = np.seterr(all='ignore')
            lars = Lars(fit_intercept=False, verbose=verbose, precompute=gram, n_nonzero_coefs=int(regularization), fit_path=False)
            lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lars.coef_
        finally:
            np.seterr(**err_mgt)
    elif algorithm == 'threshold':
        new_code = (np.sign(cov) * np.maximum(np.abs(cov) - regularization, 0)).T
        if positive:
            np.clip(new_code, 0, None, out=new_code)
    elif algorithm == 'omp':
        new_code = orthogonal_mp_gram(Gram=gram, Xy=cov, n_nonzero_coefs=int(regularization), tol=None, norms_squared=row_norms(X, squared=True), copy_Xy=copy_cov).T
    return new_code.reshape(n_samples, n_components)