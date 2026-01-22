import warnings
from math import log
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.linalg import pinvh
from ..base import RegressorMixin, _fit_context
from ..utils import _safe_indexing
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import fast_logdet
from ..utils.validation import _check_sample_weight
from ._base import LinearModel, _preprocess_data, _rescale_data
class ARDRegression(RegressorMixin, LinearModel):
    """Bayesian ARD regression.

    Fit the weights of a regression model, using an ARD prior. The weights of
    the regression model are assumed to be in Gaussian distributions.
    Also estimate the parameters lambda (precisions of the distributions of the
    weights) and alpha (precision of the distribution of the noise).
    The estimation is done by an iterative procedures (Evidence Maximization)

    Read more in the :ref:`User Guide <bayesian_regression>`.

    Parameters
    ----------
    max_iter : int, default=None
        Maximum number of iterations. If `None`, it corresponds to `max_iter=300`.

        .. versionchanged:: 1.3

    tol : float, default=1e-3
        Stop the algorithm if w has converged.

    alpha_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter.

    alpha_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter.

    lambda_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter.

    lambda_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter.

    compute_score : bool, default=False
        If True, compute the objective function at each step of the model.

    threshold_lambda : float, default=10 000
        Threshold for removing (pruning) weights with high precision from
        the computation.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    verbose : bool, default=False
        Verbose mode when fitting the model.

    n_iter : int
        Maximum number of iterations.

        .. deprecated:: 1.3
           `n_iter` is deprecated in 1.3 and will be removed in 1.5. Use
           `max_iter` instead.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of distribution)

    alpha_ : float
       estimated precision of the noise.

    lambda_ : array-like of shape (n_features,)
       estimated precisions of the weights.

    sigma_ : array-like of shape (n_features, n_features)
        estimated variance-covariance matrix of the weights

    scores_ : float
        if computed, value of the objective function (to be maximized)

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

        .. versionadded:: 1.3

    intercept_ : float
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    X_offset_ : float
        If `fit_intercept=True`, offset subtracted for centering data to a
        zero mean. Set to np.zeros(n_features) otherwise.

    X_scale_ : float
        Set to np.ones(n_features).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    BayesianRidge : Bayesian ridge regression.

    Notes
    -----
    For an example, see :ref:`examples/linear_model/plot_ard.py
    <sphx_glr_auto_examples_linear_model_plot_ard.py>`.

    References
    ----------
    D. J. C. MacKay, Bayesian nonlinear modeling for the prediction
    competition, ASHRAE Transactions, 1994.

    R. Salakhutdinov, Lecture notes on Statistical Machine Learning,
    http://www.utstat.toronto.edu/~rsalakhu/sta4273/notes/Lecture2.pdf#page=15
    Their beta is our ``self.alpha_``
    Their alpha is our ``self.lambda_``
    ARD is a little different than the slide: only dimensions/features for
    which ``self.lambda_ < self.threshold_lambda`` are kept and the rest are
    discarded.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.ARDRegression()
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    ARDRegression()
    >>> clf.predict([[1, 1]])
    array([1.])
    """
    _parameter_constraints: dict = {'max_iter': [Interval(Integral, 1, None, closed='left'), None], 'tol': [Interval(Real, 0, None, closed='left')], 'alpha_1': [Interval(Real, 0, None, closed='left')], 'alpha_2': [Interval(Real, 0, None, closed='left')], 'lambda_1': [Interval(Real, 0, None, closed='left')], 'lambda_2': [Interval(Real, 0, None, closed='left')], 'compute_score': ['boolean'], 'threshold_lambda': [Interval(Real, 0, None, closed='left')], 'fit_intercept': ['boolean'], 'copy_X': ['boolean'], 'verbose': ['verbose'], 'n_iter': [Interval(Integral, 1, None, closed='left'), Hidden(StrOptions({'deprecated'}))]}

    def __init__(self, *, max_iter=None, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000.0, fit_intercept=True, copy_X=True, verbose=False, n_iter='deprecated'):
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.compute_score = compute_score
        self.threshold_lambda = threshold_lambda
        self.copy_X = copy_X
        self.verbose = verbose
        self.n_iter = n_iter

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the model according to the given training data and parameters.

        Iterative procedure to maximize the evidence

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target values (integers). Will be cast to X's dtype if necessary.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        max_iter = _deprecate_n_iter(self.n_iter, self.max_iter)
        X, y = self._validate_data(X, y, dtype=[np.float64, np.float32], y_numeric=True, ensure_min_samples=2)
        dtype = X.dtype
        n_samples, n_features = X.shape
        coef_ = np.zeros(n_features, dtype=dtype)
        X, y, X_offset_, y_offset_, X_scale_ = _preprocess_data(X, y, fit_intercept=self.fit_intercept, copy=self.copy_X)
        self.X_offset_ = X_offset_
        self.X_scale_ = X_scale_
        keep_lambda = np.ones(n_features, dtype=bool)
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        verbose = self.verbose
        eps = np.finfo(np.float64).eps
        alpha_ = np.asarray(1.0 / (np.var(y) + eps), dtype=dtype)
        lambda_ = np.ones(n_features, dtype=dtype)
        self.scores_ = list()
        coef_old_ = None

        def update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_):
            coef_[keep_lambda] = alpha_ * np.linalg.multi_dot([sigma_, X[:, keep_lambda].T, y])
            return coef_
        update_sigma = self._update_sigma if n_samples >= n_features else self._update_sigma_woodbury
        for iter_ in range(max_iter):
            sigma_ = update_sigma(X, alpha_, lambda_, keep_lambda)
            coef_ = update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_)
            rmse_ = np.sum((y - np.dot(X, coef_)) ** 2)
            gamma_ = 1.0 - lambda_[keep_lambda] * np.diag(sigma_)
            lambda_[keep_lambda] = (gamma_ + 2.0 * lambda_1) / (coef_[keep_lambda] ** 2 + 2.0 * lambda_2)
            alpha_ = (n_samples - gamma_.sum() + 2.0 * alpha_1) / (rmse_ + 2.0 * alpha_2)
            keep_lambda = lambda_ < self.threshold_lambda
            coef_[~keep_lambda] = 0
            if self.compute_score:
                s = (lambda_1 * np.log(lambda_) - lambda_2 * lambda_).sum()
                s += alpha_1 * log(alpha_) - alpha_2 * alpha_
                s += 0.5 * (fast_logdet(sigma_) + n_samples * log(alpha_) + np.sum(np.log(lambda_)))
                s -= 0.5 * (alpha_ * rmse_ + (lambda_ * coef_ ** 2).sum())
                self.scores_.append(s)
            if iter_ > 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
                if verbose:
                    print('Converged after %s iterations' % iter_)
                break
            coef_old_ = np.copy(coef_)
            if not keep_lambda.any():
                break
        self.n_iter_ = iter_ + 1
        if keep_lambda.any():
            sigma_ = update_sigma(X, alpha_, lambda_, keep_lambda)
            coef_ = update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_)
        else:
            sigma_ = np.array([]).reshape(0, 0)
        self.coef_ = coef_
        self.alpha_ = alpha_
        self.sigma_ = sigma_
        self.lambda_ = lambda_
        self._set_intercept(X_offset_, y_offset_, X_scale_)
        return self

    def _update_sigma_woodbury(self, X, alpha_, lambda_, keep_lambda):
        n_samples = X.shape[0]
        X_keep = X[:, keep_lambda]
        inv_lambda = 1 / lambda_[keep_lambda].reshape(1, -1)
        sigma_ = pinvh(np.eye(n_samples, dtype=X.dtype) / alpha_ + np.dot(X_keep * inv_lambda, X_keep.T))
        sigma_ = np.dot(sigma_, X_keep * inv_lambda)
        sigma_ = -np.dot(inv_lambda.reshape(-1, 1) * X_keep.T, sigma_)
        sigma_[np.diag_indices(sigma_.shape[1])] += 1.0 / lambda_[keep_lambda]
        return sigma_

    def _update_sigma(self, X, alpha_, lambda_, keep_lambda):
        X_keep = X[:, keep_lambda]
        gram = np.dot(X_keep.T, X_keep)
        eye = np.eye(gram.shape[0], dtype=X.dtype)
        sigma_inv = lambda_[keep_lambda] * eye + alpha_ * gram
        sigma_ = pinvh(sigma_inv)
        return sigma_

    def predict(self, X, return_std=False):
        """Predict using the linear model.

        In addition to the mean of the predictive distribution, also its
        standard deviation can be returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        return_std : bool, default=False
            Whether to return the standard deviation of posterior prediction.

        Returns
        -------
        y_mean : array-like of shape (n_samples,)
            Mean of predictive distribution of query points.

        y_std : array-like of shape (n_samples,)
            Standard deviation of predictive distribution of query points.
        """
        y_mean = self._decision_function(X)
        if return_std is False:
            return y_mean
        else:
            col_index = self.lambda_ < self.threshold_lambda
            X = _safe_indexing(X, indices=col_index, axis=1)
            sigmas_squared_data = (np.dot(X, self.sigma_) * X).sum(axis=1)
            y_std = np.sqrt(sigmas_squared_data + 1.0 / self.alpha_)
            return (y_mean, y_std)