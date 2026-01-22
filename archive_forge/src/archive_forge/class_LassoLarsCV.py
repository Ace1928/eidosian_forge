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
class LassoLarsCV(LarsCV):
    """Cross-validated Lasso, using the LARS algorithm.

    See glossary entry for :term:`cross-validation estimator`.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    verbose : bool or int, default=False
        Sets the verbosity amount.

    max_iter : int, default=500
        Maximum number of iterations to perform.

    precompute : bool or 'auto' , default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram matrix
        cannot be passed as argument since we will use only subsets of X.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    max_n_alphas : int, default=1000
        The maximum number of points on the path used to compute the
        residuals in the cross-validation.

    n_jobs : int or None, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    positive : bool, default=False
        Restrict coefficients to be >= 0. Be aware that you might want to
        remove fit_intercept which is set True by default.
        Under the positive restriction the model coefficients do not converge
        to the ordinary-least-squares solution for small values of alpha.
        Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
        0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
        algorithm are typically in congruence with the solution of the
        coordinate descent Lasso estimator.
        As a consequence using LassoLarsCV only makes sense for problems where
        a sparse solution is expected and/or reached.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        parameter vector (w in the formulation formula)

    intercept_ : float
        independent term in decision function.

    coef_path_ : array-like of shape (n_features, n_alphas)
        the varying values of the coefficients along the path

    alpha_ : float
        the estimated regularization parameter alpha

    alphas_ : array-like of shape (n_alphas,)
        the different values of alpha along the path

    cv_alphas_ : array-like of shape (n_cv_alphas,)
        all the values of alpha along the path for the different folds

    mse_path_ : array-like of shape (n_folds, n_cv_alphas)
        the mean square error on left-out for each fold along the path
        (alpha values given by ``cv_alphas``)

    n_iter_ : array-like or int
        the number of iterations run by Lars with the optimal alpha.

    active_ : list of int
        Indices of active variables at the end of the path.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    lars_path : Compute Least Angle Regression or Lasso
        path using LARS algorithm.
    lasso_path : Compute Lasso path with coordinate descent.
    Lasso : Linear Model trained with L1 prior as
        regularizer (aka the Lasso).
    LassoCV : Lasso linear model with iterative fitting
        along a regularization path.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    LassoLarsIC : Lasso model fit with Lars using BIC
        or AIC for model selection.
    sklearn.decomposition.sparse_encode : Sparse coding.

    Notes
    -----
    The object solves the same problem as the
    :class:`~sklearn.linear_model.LassoCV` object. However, unlike the
    :class:`~sklearn.linear_model.LassoCV`, it find the relevant alphas values
    by itself. In general, because of this property, it will be more stable.
    However, it is more fragile to heavily multicollinear datasets.

    It is more efficient than the :class:`~sklearn.linear_model.LassoCV` if
    only a small number of features are selected compared to the total number,
    for instance if there are very few samples compared to the number of
    features.

    In `fit`, once the best parameter `alpha` is found through
    cross-validation, the model is fit again using the entire training set.

    Examples
    --------
    >>> from sklearn.linear_model import LassoLarsCV
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(noise=4.0, random_state=0)
    >>> reg = LassoLarsCV(cv=5).fit(X, y)
    >>> reg.score(X, y)
    0.9993...
    >>> reg.alpha_
    0.3972...
    >>> reg.predict(X[:1,])
    array([-78.4831...])
    """
    _parameter_constraints = {**LarsCV._parameter_constraints, 'positive': ['boolean']}
    method = 'lasso'

    def __init__(self, *, fit_intercept=True, verbose=False, max_iter=500, precompute='auto', cv=None, max_n_alphas=1000, n_jobs=None, eps=np.finfo(float).eps, copy_X=True, positive=False):
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.max_iter = max_iter
        self.precompute = precompute
        self.cv = cv
        self.max_n_alphas = max_n_alphas
        self.n_jobs = n_jobs
        self.eps = eps
        self.copy_X = copy_X
        self.positive = positive