import numbers
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from scipy import linalg, optimize, sparse
from scipy.sparse.linalg import lsqr
from scipy.special import expit
from ..base import (
from ..utils import check_array, check_random_state
from ..utils._array_api import get_namespace
from ..utils._seq_dataset import (
from ..utils.extmath import safe_sparse_dot
from ..utils.parallel import Parallel, delayed
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import FLOAT_DTYPES, _check_sample_weight, check_is_fitted
class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    """
    Ordinary least squares Linear Regression.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This will only provide
        speedup in case of sufficiently large problems, that is if firstly
        `n_targets > 1` and secondly `X` is sparse or if `positive` is set
        to `True`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive. This
        option is only supported for dense arrays.

        .. versionadded:: 0.24

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    rank_ : int
        Rank of matrix `X`. Only available when `X` is dense.

    singular_ : array of shape (min(X, y),)
        Singular values of `X`. Only available when `X` is dense.

    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    Ridge : Ridge regression addresses some of the
        problems of Ordinary Least Squares by imposing a penalty on the
        size of the coefficients with l2 regularization.
    Lasso : The Lasso is a linear model that estimates
        sparse coefficients with l1 regularization.
    ElasticNet : Elastic-Net is a linear regression
        model trained with both l1 and l2 -norm regularization of the
        coefficients.

    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares
    (scipy.optimize.nnls) wrapped as a predictor object.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.score(X, y)
    1.0
    >>> reg.coef_
    array([1., 2.])
    >>> reg.intercept_
    3.0...
    >>> reg.predict(np.array([[3, 5]]))
    array([16.])
    """
    _parameter_constraints: dict = {'fit_intercept': ['boolean'], 'copy_X': ['boolean'], 'n_jobs': [None, Integral], 'positive': ['boolean']}

    def __init__(self, *, fit_intercept=True, copy_X=True, n_jobs=None, positive=False):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        n_jobs_ = self.n_jobs
        accept_sparse = False if self.positive else ['csr', 'csc', 'coo']
        X, y = self._validate_data(X, y, accept_sparse=accept_sparse, y_numeric=True, multi_output=True)
        has_sw = sample_weight is not None
        if has_sw:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, only_non_negative=True)
        copy_X_in_preprocess_data = self.copy_X and (not sp.issparse(X))
        X, y, X_offset, y_offset, X_scale = _preprocess_data(X, y, fit_intercept=self.fit_intercept, copy=copy_X_in_preprocess_data, sample_weight=sample_weight)
        if has_sw:
            X, y, sample_weight_sqrt = _rescale_data(X, y, sample_weight, inplace=copy_X_in_preprocess_data)
        if self.positive:
            if y.ndim < 2:
                self.coef_ = optimize.nnls(X, y)[0]
            else:
                outs = Parallel(n_jobs=n_jobs_)((delayed(optimize.nnls)(X, y[:, j]) for j in range(y.shape[1])))
                self.coef_ = np.vstack([out[0] for out in outs])
        elif sp.issparse(X):
            X_offset_scale = X_offset / X_scale
            if has_sw:

                def matvec(b):
                    return X.dot(b) - sample_weight_sqrt * b.dot(X_offset_scale)

                def rmatvec(b):
                    return X.T.dot(b) - X_offset_scale * b.dot(sample_weight_sqrt)
            else:

                def matvec(b):
                    return X.dot(b) - b.dot(X_offset_scale)

                def rmatvec(b):
                    return X.T.dot(b) - X_offset_scale * b.sum()
            X_centered = sparse.linalg.LinearOperator(shape=X.shape, matvec=matvec, rmatvec=rmatvec)
            if y.ndim < 2:
                self.coef_ = lsqr(X_centered, y)[0]
            else:
                outs = Parallel(n_jobs=n_jobs_)((delayed(lsqr)(X_centered, y[:, j].ravel()) for j in range(y.shape[1])))
                self.coef_ = np.vstack([out[0] for out in outs])
        else:
            self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)
            self.coef_ = self.coef_.T
        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self