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
class LassoCV(RegressorMixin, LinearModelCV):
    """Lasso linear model with iterative fitting along a regularization path.

    See glossary entry for :term:`cross-validation estimator`.

    The best model is selected by cross-validation.

    The optimization objective for Lasso is::

        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Read more in the :ref:`User Guide <lasso>`.

    Parameters
    ----------
    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path.

    alphas : array-like, default=None
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically.

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

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

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

    verbose : bool or int, default=False
        Amount of verbosity.

    n_jobs : int, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    positive : bool, default=False
        If positive, restrict regression coefficients to be positive.

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

    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the cost function formula).

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function.

    mse_path_ : ndarray of shape (n_alphas, n_folds)
        Mean square error for the test set on each fold, varying alpha.

    alphas_ : ndarray of shape (n_alphas,)
        The grid of alphas used for fitting.

    dual_gap_ : float or ndarray of shape (n_targets,)
        The dual gap at the end of the optimization for the optimal alpha
        (``alpha_``).

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
    lars_path : Compute Least Angle Regression or Lasso path using LARS
        algorithm.
    lasso_path : Compute Lasso path with coordinate descent.
    Lasso : The Lasso is a linear model that estimates sparse coefficients.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    LassoCV : Lasso linear model with iterative fitting along a regularization
        path.
    LassoLarsCV : Cross-validated Lasso using the LARS algorithm.

    Notes
    -----
    In `fit`, once the best parameter `alpha` is found through
    cross-validation, the model is fit again using the entire training set.

    To avoid unnecessary memory duplication the `X` argument of the `fit`
    method should be directly passed as a Fortran-contiguous numpy array.

     For an example, see
     :ref:`examples/linear_model/plot_lasso_model_selection.py
     <sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py>`.

    :class:`LassoCV` leads to different results than a hyperparameter
    search using :class:`~sklearn.model_selection.GridSearchCV` with a
    :class:`Lasso` model. In :class:`LassoCV`, a model for a given
    penalty `alpha` is warm started using the coefficients of the
    closest model (trained at the previous iteration) on the
    regularization path. It tends to speed up the hyperparameter
    search.

    Examples
    --------
    >>> from sklearn.linear_model import LassoCV
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(noise=4, random_state=0)
    >>> reg = LassoCV(cv=5, random_state=0).fit(X, y)
    >>> reg.score(X, y)
    0.9993...
    >>> reg.predict(X[:1,])
    array([-78.4951...])
    """
    path = staticmethod(lasso_path)

    def __init__(self, *, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, precompute='auto', max_iter=1000, tol=0.0001, copy_X=True, cv=None, verbose=False, n_jobs=None, positive=False, random_state=None, selection='cyclic'):
        super().__init__(eps=eps, n_alphas=n_alphas, alphas=alphas, fit_intercept=fit_intercept, precompute=precompute, max_iter=max_iter, tol=tol, copy_X=copy_X, cv=cv, verbose=verbose, n_jobs=n_jobs, positive=positive, random_state=random_state, selection=selection)

    def _get_estimator(self):
        return Lasso()

    def _is_multitask(self):
        return False

    def _more_tags(self):
        return {'multioutput': False}