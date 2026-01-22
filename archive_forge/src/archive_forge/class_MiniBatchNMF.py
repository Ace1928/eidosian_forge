import itertools
import time
import warnings
from abc import ABC
from math import sqrt
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from .._config import config_context
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_random_state, gen_batches, metadata_routing
from ..utils._param_validation import (
from ..utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from ..utils.validation import (
from ._cdnmf_fast import _update_cdnmf_fast
class MiniBatchNMF(_BaseNMF):
    """Mini-Batch Non-Negative Matrix Factorization (NMF).

    .. versionadded:: 1.1

    Find two non-negative matrices, i.e. matrices with all non-negative elements,
    (`W`, `H`) whose product approximates the non-negative matrix `X`. This
    factorization can be used for example for dimensionality reduction, source
    separation or topic extraction.

    The objective function is:

        .. math::

            L(W, H) &= 0.5 * ||X - WH||_{loss}^2

            &+ alpha\\_W * l1\\_ratio * n\\_features * ||vec(W)||_1

            &+ alpha\\_H * l1\\_ratio * n\\_samples * ||vec(H)||_1

            &+ 0.5 * alpha\\_W * (1 - l1\\_ratio) * n\\_features * ||W||_{Fro}^2

            &+ 0.5 * alpha\\_H * (1 - l1\\_ratio) * n\\_samples * ||H||_{Fro}^2

    Where:

    :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm)

    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm)

    The generic norm :math:`||X - WH||_{loss}^2` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.

    The objective function is minimized with an alternating minimization of `W`
    and `H`.

    Note that the transformed data is named `W` and the components matrix is
    named `H`. In the NMF literature, the naming convention is usually the opposite
    since the data matrix `X` is transposed.

    Read more in the :ref:`User Guide <MiniBatchNMF>`.

    Parameters
    ----------
    n_components : int or {'auto'} or None, default=None
        Number of components, if `n_components` is not set all features
        are kept.
        If `n_components='auto'`, the number of components is automatically inferred
        from W or H shapes.

        .. versionchanged:: 1.4
            Added `'auto'` value.

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - `None`: 'nndsvda' if `n_components <= min(n_samples, n_features)`,
          otherwise random.

        - `'random'`: non-negative random matrices, scaled with:
          `sqrt(X.mean() / n_components)`

        - `'nndsvd'`: Nonnegative Double Singular Value Decomposition (NNDSVD)
          initialization (better for sparseness).

        - `'nndsvda'`: NNDSVD with zeros filled with the average of X
          (better when sparsity is not desired).

        - `'nndsvdar'` NNDSVD with zeros filled with small random values
          (generally faster, less accurate alternative to NNDSVDa
          for when sparsity is not desired).

        - `'custom'`: Use custom matrices `W` and `H` which must both be provided.

    batch_size : int, default=1024
        Number of samples in each mini-batch. Large batch sizes
        give better long-term convergence at the cost of a slower start.

    beta_loss : float or {'frobenius', 'kullback-leibler',             'itakura-saito'}, default='frobenius'
        Beta divergence to be minimized, measuring the distance between `X`
        and the dot product `WH`. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for `beta_loss <= 0` (or 'itakura-saito'), the input
        matrix `X` cannot contain zeros.

    tol : float, default=1e-4
        Control early stopping based on the norm of the differences in `H`
        between 2 steps. To disable early stopping based on changes in `H`, set
        `tol` to 0.0.

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini batches
        that does not yield an improvement on the smoothed cost function.
        To disable convergence detection based on cost function, set
        `max_no_improvement` to None.

    max_iter : int, default=200
        Maximum number of iterations over the complete dataset before
        timing out.

    alpha_W : float, default=0.0
        Constant that multiplies the regularization terms of `W`. Set it to zero
        (default) to have no regularization on `W`.

    alpha_H : float or "same", default="same"
        Constant that multiplies the regularization terms of `H`. Set it to zero to
        have no regularization on `H`. If "same" (default), it takes the same value as
        `alpha_W`.

    l1_ratio : float, default=0.0
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    forget_factor : float, default=0.7
        Amount of rescaling of past information. Its value could be 1 with
        finite datasets. Choosing values < 1 is recommended with online
        learning as more recent batches will weight more than past batches.

    fresh_restarts : bool, default=False
        Whether to completely solve for W at each step. Doing fresh restarts will likely
        lead to a better solution for a same number of iterations but it is much slower.

    fresh_restarts_max_iter : int, default=30
        Maximum number of iterations when solving for W at each step. Only used when
        doing fresh restarts. These iterations may be stopped early based on a small
        change of W controlled by `tol`.

    transform_max_iter : int, default=None
        Maximum number of iterations when solving for W at transform time.
        If None, it defaults to `max_iter`.

    random_state : int, RandomState instance or None, default=None
        Used for initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : bool, default=False
        Whether to be verbose.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Factorization matrix, sometimes called 'dictionary'.

    n_components_ : int
        The number of components. It is same as the `n_components` parameter
        if it was given. Otherwise, it will be same as the number of
        features.

    reconstruction_err_ : float
        Frobenius norm of the matrix difference, or beta-divergence, between
        the training data `X` and the reconstructed data `WH` from
        the fitted model.

    n_iter_ : int
        Actual number of started iterations over the whole dataset.

    n_steps_ : int
        Number of mini-batches processed.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    NMF : Non-negative matrix factorization.
    MiniBatchDictionaryLearning : Finds a dictionary that can best be used to represent
        data using a sparse code.

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
       factorizations" <10.1587/transfun.E92.A.708>`
       Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
       of electronics, communications and computer sciences 92.3: 708-721, 2009.

    .. [2] :doi:`"Algorithms for nonnegative matrix factorization with the
       beta-divergence" <10.1162/NECO_a_00168>`
       Fevotte, C., & Idier, J. (2011). Neural Computation, 23(9).

    .. [3] :doi:`"Online algorithms for nonnegative matrix factorization with the
       Itakura-Saito divergence" <10.1109/ASPAA.2011.6082314>`
       Lefevre, A., Bach, F., Fevotte, C. (2011). WASPA.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import MiniBatchNMF
    >>> model = MiniBatchNMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_
    """
    _parameter_constraints: dict = {**_BaseNMF._parameter_constraints, 'max_no_improvement': [Interval(Integral, 1, None, closed='left'), None], 'batch_size': [Interval(Integral, 1, None, closed='left')], 'forget_factor': [Interval(Real, 0, 1, closed='both')], 'fresh_restarts': ['boolean'], 'fresh_restarts_max_iter': [Interval(Integral, 1, None, closed='left')], 'transform_max_iter': [Interval(Integral, 1, None, closed='left'), None]}

    def __init__(self, n_components='warn', *, init=None, batch_size=1024, beta_loss='frobenius', tol=0.0001, max_no_improvement=10, max_iter=200, alpha_W=0.0, alpha_H='same', l1_ratio=0.0, forget_factor=0.7, fresh_restarts=False, fresh_restarts_max_iter=30, transform_max_iter=None, random_state=None, verbose=0):
        super().__init__(n_components=n_components, init=init, beta_loss=beta_loss, tol=tol, max_iter=max_iter, random_state=random_state, alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio, verbose=verbose)
        self.max_no_improvement = max_no_improvement
        self.batch_size = batch_size
        self.forget_factor = forget_factor
        self.fresh_restarts = fresh_restarts
        self.fresh_restarts_max_iter = fresh_restarts_max_iter
        self.transform_max_iter = transform_max_iter

    def _check_params(self, X):
        super()._check_params(X)
        self._batch_size = min(self.batch_size, X.shape[0])
        self._rho = self.forget_factor ** (self._batch_size / X.shape[0])
        if self._beta_loss < 1:
            self._gamma = 1.0 / (2.0 - self._beta_loss)
        elif self._beta_loss > 2:
            self._gamma = 1.0 / (self._beta_loss - 1.0)
        else:
            self._gamma = 1.0
        self._transform_max_iter = self.max_iter if self.transform_max_iter is None else self.transform_max_iter
        return self

    def _solve_W(self, X, H, max_iter):
        """Minimize the objective function w.r.t W.

        Update W with H being fixed, until convergence. This is the heart
        of `transform` but it's also used during `fit` when doing fresh restarts.
        """
        avg = np.sqrt(X.mean() / self._n_components)
        W = np.full((X.shape[0], self._n_components), avg, dtype=X.dtype)
        W_buffer = W.copy()
        l1_reg_W, _, l2_reg_W, _ = self._compute_regularization(X)
        for _ in range(max_iter):
            W, *_ = _multiplicative_update_w(X, W, H, self._beta_loss, l1_reg_W, l2_reg_W, self._gamma)
            W_diff = linalg.norm(W - W_buffer) / linalg.norm(W)
            if self.tol > 0 and W_diff <= self.tol:
                break
            W_buffer[:] = W
        return W

    def _minibatch_step(self, X, W, H, update_H):
        """Perform the update of W and H for one minibatch."""
        batch_size = X.shape[0]
        l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = self._compute_regularization(X)
        if self.fresh_restarts or W is None:
            W = self._solve_W(X, H, self.fresh_restarts_max_iter)
        else:
            W, *_ = _multiplicative_update_w(X, W, H, self._beta_loss, l1_reg_W, l2_reg_W, self._gamma)
        if self._beta_loss < 1:
            W[W < np.finfo(np.float64).eps] = 0.0
        batch_cost = (_beta_divergence(X, W, H, self._beta_loss) + l1_reg_W * W.sum() + l1_reg_H * H.sum() + l2_reg_W * (W ** 2).sum() + l2_reg_H * (H ** 2).sum()) / batch_size
        if update_H:
            H[:] = _multiplicative_update_h(X, W, H, beta_loss=self._beta_loss, l1_reg_H=l1_reg_H, l2_reg_H=l2_reg_H, gamma=self._gamma, A=self._components_numerator, B=self._components_denominator, rho=self._rho)
            if self._beta_loss <= 1:
                H[H < np.finfo(np.float64).eps] = 0.0
        return batch_cost

    def _minibatch_convergence(self, X, batch_cost, H, H_buffer, n_samples, step, n_steps):
        """Helper function to encapsulate the early stopping logic"""
        batch_size = X.shape[0]
        step = step + 1
        if step == 1:
            if self.verbose:
                print(f'Minibatch step {step}/{n_steps}: mean batch cost: {batch_cost}')
            return False
        if self._ewa_cost is None:
            self._ewa_cost = batch_cost
        else:
            alpha = batch_size / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_cost = self._ewa_cost * (1 - alpha) + batch_cost * alpha
        if self.verbose:
            print(f'Minibatch step {step}/{n_steps}: mean batch cost: {batch_cost}, ewa cost: {self._ewa_cost}')
        H_diff = linalg.norm(H - H_buffer) / linalg.norm(H)
        if self.tol > 0 and H_diff <= self.tol:
            if self.verbose:
                print(f'Converged (small H change) at step {step}/{n_steps}')
            return True
        if self._ewa_cost_min is None or self._ewa_cost < self._ewa_cost_min:
            self._no_improvement = 0
            self._ewa_cost_min = self._ewa_cost
        else:
            self._no_improvement += 1
        if self.max_no_improvement is not None and self._no_improvement >= self.max_no_improvement:
            if self.verbose:
                print(f'Converged (lack of improvement in objective function) at step {step}/{n_steps}')
            return True
        return False

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, W=None, H=None):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed.

        y : Ignored
            Not used, present here for API consistency by convention.

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialisation method specified in `init`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialisation method specified in `init`.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        X = self._validate_data(X, accept_sparse=('csr', 'csc'), dtype=[np.float64, np.float32])
        with config_context(assume_finite=True):
            W, H, n_iter, n_steps = self._fit_transform(X, W=W, H=H)
        self.reconstruction_err_ = _beta_divergence(X, W, H, self._beta_loss, square_root=True)
        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter
        self.n_steps_ = n_steps
        return W

    def _fit_transform(self, X, W=None, H=None, update_H=True):
        """Learn a NMF model for the data X and returns the transformed data.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed.

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `update_H=False`, it is initialised as an array of zeros, unless
            `solver='mu'`, then it is filled with values calculated by
            `np.sqrt(X.mean() / self._n_components)`.
            If `None`, uses the initialisation method specified in `init`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `update_H=False`, it is used as a constant, to solve for W only.
            If `None`, uses the initialisation method specified in `init`.

        update_H : bool, default=True
            If True, both W and H will be estimated from initial guesses,
            this corresponds to a call to the `fit_transform` method.
            If False, only W will be estimated, this corresponds to a call
            to the `transform` method.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.

        H : ndarray of shape (n_components, n_features)
            Factorization matrix, sometimes called 'dictionary'.

        n_iter : int
            Actual number of started iterations over the whole dataset.

        n_steps : int
            Number of mini-batches processed.
        """
        check_non_negative(X, 'MiniBatchNMF (input X)')
        self._check_params(X)
        if X.min() == 0 and self._beta_loss <= 0:
            raise ValueError('When beta_loss <= 0 and X contains zeros, the solver may diverge. Please add small values to X, or use a positive beta_loss.')
        n_samples = X.shape[0]
        W, H = self._check_w_h(X, W, H, update_H)
        H_buffer = H.copy()
        self._components_numerator = H.copy()
        self._components_denominator = np.ones(H.shape, dtype=H.dtype)
        self._ewa_cost = None
        self._ewa_cost_min = None
        self._no_improvement = 0
        batches = gen_batches(n_samples, self._batch_size)
        batches = itertools.cycle(batches)
        n_steps_per_iter = int(np.ceil(n_samples / self._batch_size))
        n_steps = self.max_iter * n_steps_per_iter
        for i, batch in zip(range(n_steps), batches):
            batch_cost = self._minibatch_step(X[batch], W[batch], H, update_H)
            if update_H and self._minibatch_convergence(X[batch], batch_cost, H, H_buffer, n_samples, i, n_steps):
                break
            H_buffer[:] = H
        if self.fresh_restarts:
            W = self._solve_W(X, H, self._transform_max_iter)
        n_steps = i + 1
        n_iter = int(np.ceil(n_steps / n_steps_per_iter))
        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(f'Maximum number of iterations {self.max_iter} reached. Increase it to improve convergence.', ConvergenceWarning)
        return (W, H, n_iter, n_steps)

    def transform(self, X):
        """Transform the data X according to the fitted MiniBatchNMF model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be transformed by the model.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=('csr', 'csc'), dtype=[np.float64, np.float32], reset=False)
        W = self._solve_W(X, self.components_, self._transform_max_iter)
        return W

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None, W=None, H=None):
        """Update the model using the data in `X` as a mini-batch.

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once (see :ref:`scaling_strategies`).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed.

        y : Ignored
            Not used, present here for API consistency by convention.

        W : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            Only used for the first call to `partial_fit`.

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            Only used for the first call to `partial_fit`.

        Returns
        -------
        self
            Returns the instance itself.
        """
        has_components = hasattr(self, 'components_')
        X = self._validate_data(X, accept_sparse=('csr', 'csc'), dtype=[np.float64, np.float32], reset=not has_components)
        if not has_components:
            self._check_params(X)
            _, H = self._check_w_h(X, W=W, H=H, update_H=True)
            self._components_numerator = H.copy()
            self._components_denominator = np.ones(H.shape, dtype=H.dtype)
            self.n_steps_ = 0
        else:
            H = self.components_
        self._minibatch_step(X, None, H, update_H=True)
        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_steps_ += 1
        return self