from numbers import Integral, Real
import numpy as np
import scipy.optimize
from ..._loss.loss import (
from ...base import BaseEstimator, RegressorMixin, _fit_context
from ...utils import check_array
from ...utils._openmp_helpers import _openmp_effective_n_threads
from ...utils._param_validation import Hidden, Interval, StrOptions
from ...utils.optimize import _check_optimize_result
from ...utils.validation import _check_sample_weight, check_is_fitted
from .._linear_loss import LinearModelLoss
from ._newton_solver import NewtonCholeskySolver, NewtonSolver
class GammaRegressor(_GeneralizedLinearRegressor):
    """Generalized Linear Model with a Gamma distribution.

    This regressor uses the 'log' link function.

    Read more in the :ref:`User Guide <Generalized_linear_models>`.

    .. versionadded:: 0.23

    Parameters
    ----------
    alpha : float, default=1
        Constant that multiplies the L2 penalty term and determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).
        Values of `alpha` must be in the range `[0.0, inf)`.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor `X @ coef_ + intercept_`.

    solver : {'lbfgs', 'newton-cholesky'}, default='lbfgs'
        Algorithm to use in the optimization problem:

        'lbfgs'
            Calls scipy's L-BFGS-B optimizer.

        'newton-cholesky'
            Uses Newton-Raphson steps (in arbitrary precision arithmetic equivalent to
            iterated reweighted least squares) with an inner Cholesky based solver.
            This solver is a good choice for `n_samples` >> `n_features`, especially
            with one-hot encoded categorical features with rare categories. Be aware
            that the memory usage of this solver has a quadratic dependency on
            `n_features` because it explicitly computes the Hessian matrix.

            .. versionadded:: 1.2

    max_iter : int, default=100
        The maximal number of iterations for the solver.
        Values must be in the range `[1, inf)`.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.
        Values must be in the range `(0.0, inf)`.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for `coef_` and `intercept_`.

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.
        Values must be in the range `[0, inf)`.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    n_iter_ : int
        Actual number of iterations used in the solver.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PoissonRegressor : Generalized Linear Model with a Poisson distribution.
    TweedieRegressor : Generalized Linear Model with a Tweedie distribution.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.GammaRegressor()
    >>> X = [[1, 2], [2, 3], [3, 4], [4, 3]]
    >>> y = [19, 26, 33, 30]
    >>> clf.fit(X, y)
    GammaRegressor()
    >>> clf.score(X, y)
    0.773...
    >>> clf.coef_
    array([0.072..., 0.066...])
    >>> clf.intercept_
    2.896...
    >>> clf.predict([[1, 0], [2, 8]])
    array([19.483..., 35.795...])
    """
    _parameter_constraints: dict = {**_GeneralizedLinearRegressor._parameter_constraints}

    def __init__(self, *, alpha=1.0, fit_intercept=True, solver='lbfgs', max_iter=100, tol=0.0001, warm_start=False, verbose=0):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, solver=solver, max_iter=max_iter, tol=tol, warm_start=warm_start, verbose=verbose)

    def _get_loss(self):
        return HalfGammaLoss()