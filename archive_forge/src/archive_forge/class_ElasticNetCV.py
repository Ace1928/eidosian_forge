import numbers
import sys
import warnings
from abc import ABC, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy import sparse
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..model_selection import check_cv
from ..utils import Bunch, check_array, check_scalar
from ..utils._metadata_requests import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import safe_sparse_dot
from ..utils.metadata_routing import (
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from . import _cd_fast as cd_fast  # type: ignore
from ._base import LinearModel, _pre_fit, _preprocess_data
class ElasticNetCV(RegressorMixin, LinearModelCV):
    """Elastic Net model with iterative fitting along a regularization path.

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <elastic_net>`.

    Parameters
    ----------
    l1_ratio : float or list of float, default=0.5
        Float between 0 and 1 passed to ElasticNet (scaling between
        l1 and l2 penalties). For ``l1_ratio = 0``
        the penalty is an L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2
        This parameter can be a list, in which case the different
        values are tested by cross-validation and the one giving the best
        prediction score is used. Note that a good choice of list of
        values for l1_ratio is often to put more values close to 1
        (i.e. Lasso) and less close to 0 (i.e. Ridge), as in ``[.1, .5, .7,
        .9, .95, .99, 1]``.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path, used for each l1_ratio.

    alphas : array-like, default=None
        List of alphas where to compute the models.
        If None alphas are set automatically.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    precompute : 'auto', bool or array-like of shape             (n_features, n_features), default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    verbose : bool or int, default=0
        Amount of verbosity.

    n_jobs : int, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    alpha_ : float
        The amount of penalization chosen by cross validation.

    l1_ratio_ : float
        The compromise between l1 and l2 penalization chosen by
        cross validation.

    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the cost function formula).

    intercept_ : float or ndarray of shape (n_targets, n_features)
        Independent term in the decision function.

    mse_path_ : ndarray of shape (n_l1_ratio, n_alpha, n_folds)
        Mean square error for the test set on each fold, varying l1_ratio and
        alpha.

    alphas_ : ndarray of shape (n_alphas,) or (n_l1_ratio, n_alphas)
        The grid of alphas used for fitting, for each l1_ratio.

    dual_gap_ : float
        The dual gaps at the end of the optimization for the optimal alpha.

    n_iter_ : int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    enet_path : Compute elastic net path with coordinate descent.
    ElasticNet : Linear regression with combined L1 and L2 priors as regularizer.

    Notes
    -----
    In `fit`, once the best parameters `l1_ratio` and `alpha` are found through
    cross-validation, the model is fit again using the entire training set.

    To avoid unnecessary memory duplication the `X` argument of the `fit`
    method should be directly passed as a Fortran-contiguous numpy array.

    The parameter `l1_ratio` corresponds to alpha in the glmnet R package
    while alpha corresponds to the lambda parameter in glmnet.
    More specifically, the optimization objective is::

        1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    If you are interested in controlling the L1 and L2 penalty
    separately, keep in mind that this is equivalent to::

        a * L1 + b * L2

    for::

        alpha = a + b and l1_ratio = a / (a + b).

    For an example, see
    :ref:`examples/linear_model/plot_lasso_model_selection.py
    <sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py>`.

    Examples
    --------
    >>> from sklearn.linear_model import ElasticNetCV
    >>> from sklearn.datasets import make_regression

    >>> X, y = make_regression(n_features=2, random_state=0)
    >>> regr = ElasticNetCV(cv=5, random_state=0)
    >>> regr.fit(X, y)
    ElasticNetCV(cv=5, random_state=0)
    >>> print(regr.alpha_)
    0.199...
    >>> print(regr.intercept_)
    0.398...
    >>> print(regr.predict([[0, 0]]))
    [0.398...]
    """
    _parameter_constraints: dict = {**LinearModelCV._parameter_constraints, 'l1_ratio': [Interval(Real, 0, 1, closed='both'), 'array-like']}
    path = staticmethod(enet_path)

    def __init__(self, *, l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, precompute='auto', max_iter=1000, tol=0.0001, cv=None, copy_X=True, verbose=0, n_jobs=None, positive=False, random_state=None, selection='cyclic'):
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.copy_X = copy_X
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    def _get_estimator(self):
        return ElasticNet()

    def _is_multitask(self):
        return False

    def _more_tags(self):
        return {'multioutput': False}