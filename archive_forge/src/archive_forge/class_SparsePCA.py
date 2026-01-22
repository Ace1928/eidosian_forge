from numbers import Integral, Real
import numpy as np
from ..base import (
from ..linear_model import ridge_regression
from ..utils import check_random_state
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import svd_flip
from ..utils.validation import check_array, check_is_fitted
from ._dict_learning import MiniBatchDictionaryLearning, dict_learning
class SparsePCA(_BaseSparsePCA):
    """Sparse Principal Components Analysis (SparsePCA).

    Finds the set of sparse components that can optimally reconstruct
    the data.  The amount of sparseness is controllable by the coefficient
    of the L1 penalty, given by the parameter alpha.

    Read more in the :ref:`User Guide <SparsePCA>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of sparse atoms to extract. If None, then ``n_components``
        is set to ``n_features``.

    alpha : float, default=1
        Sparsity controlling parameter. Higher values lead to sparser
        components.

    ridge_alpha : float, default=0.01
        Amount of ridge shrinkage to apply in order to improve
        conditioning when calling the transform method.

    max_iter : int, default=1000
        Maximum number of iterations to perform.

    tol : float, default=1e-8
        Tolerance for the stopping condition.

    method : {'lars', 'cd'}, default='lars'
        Method to be used for optimization.
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    U_init : ndarray of shape (n_samples, n_components), default=None
        Initial values for the loadings for warm restart scenarios. Only used
        if `U_init` and `V_init` are not None.

    V_init : ndarray of shape (n_components, n_features), default=None
        Initial values for the components for warm restart scenarios. Only used
        if `U_init` and `V_init` are not None.

    verbose : int or bool, default=False
        Controls the verbosity; the higher, the more messages. Defaults to 0.

    random_state : int, RandomState instance or None, default=None
        Used during dictionary learning. Pass an int for reproducible results
        across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Sparse components extracted from the data.

    error_ : ndarray
        Vector of errors at each iteration.

    n_components_ : int
        Estimated number of components.

        .. versionadded:: 0.23

    n_iter_ : int
        Number of iterations run.

    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
        Equal to ``X.mean(axis=0)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PCA : Principal Component Analysis implementation.
    MiniBatchSparsePCA : Mini batch variant of `SparsePCA` that is faster but less
        accurate.
    DictionaryLearning : Generic dictionary learning problem using a sparse code.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.decomposition import SparsePCA
    >>> X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
    >>> transformer = SparsePCA(n_components=5, random_state=0)
    >>> transformer.fit(X)
    SparsePCA(...)
    >>> X_transformed = transformer.transform(X)
    >>> X_transformed.shape
    (200, 5)
    >>> # most values in the components_ are zero (sparsity)
    >>> np.mean(transformer.components_ == 0)
    0.9666...
    """
    _parameter_constraints: dict = {**_BaseSparsePCA._parameter_constraints, 'U_init': [None, np.ndarray], 'V_init': [None, np.ndarray]}

    def __init__(self, n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-08, method='lars', n_jobs=None, U_init=None, V_init=None, verbose=False, random_state=None):
        super().__init__(n_components=n_components, alpha=alpha, ridge_alpha=ridge_alpha, max_iter=max_iter, tol=tol, method=method, n_jobs=n_jobs, verbose=verbose, random_state=random_state)
        self.U_init = U_init
        self.V_init = V_init

    def _fit(self, X, n_components, random_state):
        """Specialized `fit` for SparsePCA."""
        code_init = self.V_init.T if self.V_init is not None else None
        dict_init = self.U_init.T if self.U_init is not None else None
        code, dictionary, E, self.n_iter_ = dict_learning(X.T, n_components, alpha=self.alpha, tol=self.tol, max_iter=self.max_iter, method=self.method, n_jobs=self.n_jobs, verbose=self.verbose, random_state=random_state, code_init=code_init, dict_init=dict_init, return_n_iter=True)
        code, dictionary = svd_flip(code, dictionary, u_based_decision=False)
        self.components_ = code.T
        components_norm = np.linalg.norm(self.components_, axis=1)[:, np.newaxis]
        components_norm[components_norm == 0] = 1
        self.components_ /= components_norm
        self.n_components_ = len(self.components_)
        self.error_ = E
        return self