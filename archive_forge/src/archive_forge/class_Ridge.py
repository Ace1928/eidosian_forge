import numbers
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from scipy import linalg, optimize, sparse
from scipy.sparse import linalg as sp_linalg
from ..base import MultiOutputMixin, RegressorMixin, _fit_context, is_classifier
from ..exceptions import ConvergenceWarning
from ..metrics import check_scoring, get_scorer_names
from ..model_selection import GridSearchCV
from ..preprocessing import LabelBinarizer
from ..utils import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import _sparse_linalg_cg
from ..utils.metadata_routing import (
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, LinearModel, _preprocess_data, _rescale_data
from ._sag import sag_solver
class Ridge(MultiOutputMixin, RegressorMixin, _BaseRidge):
    """Linear least squares with l2 regularization.

    Minimizes the objective function::

    ||y - Xw||^2_2 + alpha * ||w||^2_2

    This model solves a regression model where the loss function is
    the linear least squares function and regularization is given by
    the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
    This estimator has built-in support for multi-variate regression
    (i.e., when y is a 2d-array of shape (n_samples, n_targets)).

    Read more in the :ref:`User Guide <ridge_regression>`.

    Parameters
    ----------
    alpha : {float, ndarray of shape (n_targets,)}, default=1.0
        Constant that multiplies the L2 term, controlling regularization
        strength. `alpha` must be a non-negative float i.e. in `[0, inf)`.

        When `alpha = 0`, the objective is equivalent to ordinary least
        squares, solved by the :class:`LinearRegression` object. For numerical
        reasons, using `alpha = 0` with the `Ridge` object is not advised.
        Instead, you should use the :class:`LinearRegression` object.

        If an array is passed, penalties are assumed to be specific to the
        targets. Hence they must correspond in number.

    fit_intercept : bool, default=True
        Whether to fit the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. ``X`` and ``y`` are expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    max_iter : int, default=None
        Maximum number of iterations for conjugate gradient solver.
        For 'sparse_cg' and 'lsqr' solvers, the default value is determined
        by scipy.sparse.linalg. For 'sag' solver, the default value is 1000.
        For 'lbfgs' solver, the default value is 15000.

    tol : float, default=1e-4
        The precision of the solution (`coef_`) is determined by `tol` which
        specifies a different convergence criterion for each solver:

        - 'svd': `tol` has no impact.

        - 'cholesky': `tol` has no impact.

        - 'sparse_cg': norm of residuals smaller than `tol`.

        - 'lsqr': `tol` is set as atol and btol of scipy.sparse.linalg.lsqr,
          which control the norm of the residual vector in terms of the norms of
          matrix and coefficients.

        - 'sag' and 'saga': relative change of coef smaller than `tol`.

        - 'lbfgs': maximum of the absolute (projected) gradient=max|residuals|
          smaller than `tol`.

        .. versionchanged:: 1.2
           Default value changed from 1e-3 to 1e-4 for consistency with other linear
           models.

    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg',             'sag', 'saga', 'lbfgs'}, default='auto'
        Solver to use in the computational routines:

        - 'auto' chooses the solver automatically based on the type of data.

        - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
          coefficients. It is the most stable solver, in particular more stable
          for singular matrices than 'cholesky' at the cost of being slower.

        - 'cholesky' uses the standard scipy.linalg.solve function to
          obtain a closed-form solution.

        - 'sparse_cg' uses the conjugate gradient solver as found in
          scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
          more appropriate than 'cholesky' for large-scale data
          (possibility to set `tol` and `max_iter`).

        - 'lsqr' uses the dedicated regularized least-squares routine
          scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative
          procedure.

        - 'sag' uses a Stochastic Average Gradient descent, and 'saga' uses
          its improved, unbiased version named SAGA. Both methods also use an
          iterative procedure, and are often faster than other solvers when
          both n_samples and n_features are large. Note that 'sag' and
          'saga' fast convergence is only guaranteed on features with
          approximately the same scale. You can preprocess the data with a
          scaler from sklearn.preprocessing.

        - 'lbfgs' uses L-BFGS-B algorithm implemented in
          `scipy.optimize.minimize`. It can be used only when `positive`
          is True.

        All solvers except 'svd' support both dense and sparse data. However, only
        'lsqr', 'sag', 'sparse_cg', and 'lbfgs' support sparse input when
        `fit_intercept` is True.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.
        Only 'lbfgs' solver is supported in this case.

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag' or 'saga' to shuffle the data.
        See :term:`Glossary <random_state>` for details.

        .. versionadded:: 0.17
           `random_state` to support Stochastic Average Gradient.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    n_iter_ : None or ndarray of shape (n_targets,)
        Actual number of iterations for each target. Available only for
        sag and lsqr solvers. Other solvers will return None.

        .. versionadded:: 0.17

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    RidgeClassifier : Ridge classifier.
    RidgeCV : Ridge regression with built-in cross validation.
    :class:`~sklearn.kernel_ridge.KernelRidge` : Kernel ridge regression
        combines ridge regression with the kernel trick.

    Notes
    -----
    Regularization improves the conditioning of the problem and
    reduces the variance of the estimates. Larger values specify stronger
    regularization. Alpha corresponds to ``1 / (2C)`` in other linear
    models such as :class:`~sklearn.linear_model.LogisticRegression` or
    :class:`~sklearn.svm.LinearSVC`.

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> clf = Ridge(alpha=1.0)
    >>> clf.fit(X, y)
    Ridge()
    """

    def __init__(self, alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, copy_X=copy_X, max_iter=max_iter, tol=tol, solver=solver, positive=positive, random_state=random_state)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        _accept_sparse = _get_valid_accept_sparse(sparse.issparse(X), self.solver)
        X, y = self._validate_data(X, y, accept_sparse=_accept_sparse, dtype=[np.float64, np.float32], multi_output=True, y_numeric=True)
        return super().fit(X, y, sample_weight=sample_weight)