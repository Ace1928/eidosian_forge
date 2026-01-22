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
class MiniBatchDictionaryLearning(_BaseSparseCoding, BaseEstimator):
    """Mini-batch dictionary learning.

    Finds a dictionary (a set of atoms) that performs well at sparsely
    encoding the fitted data.

    Solves the optimization problem::

       (U^*,V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                    (U,V)
                    with || V_k ||_2 <= 1 for all  0 <= k < n_components

    ||.||_Fro stands for the Frobenius norm and ||.||_1,1 stands for
    the entry-wise matrix norm which is the sum of the absolute values
    of all the entries in the matrix.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of dictionary elements to extract.

    alpha : float, default=1
        Sparsity controlling parameter.

    max_iter : int, default=1_000
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

        .. versionadded:: 1.1

        .. deprecated:: 1.4
           `max_iter=None` is deprecated in 1.4 and will be removed in 1.6.
           Use the default value (i.e. `1_000`) instead.

    fit_algorithm : {'lars', 'cd'}, default='lars'
        The algorithm used:

        - `'lars'`: uses the least angle regression method to solve the lasso
          problem (`linear_model.lars_path`)
        - `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    batch_size : int, default=256
        Number of samples in each mini-batch.

        .. versionchanged:: 1.3
           The default value of `batch_size` changed from 3 to 256 in version 1.3.

    shuffle : bool, default=True
        Whether to shuffle the samples before forming batches.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial value of the dictionary for warm restart scenarios.

    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp',             'threshold'}, default='omp'
        Algorithm used to transform the data:

        - `'lars'`: uses the least angle regression method
          (`linear_model.lars_path`);
        - `'lasso_lars'`: uses Lars to compute the Lasso solution.
        - `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). `'lasso_lars'` will be faster
          if the estimated components are sparse.
        - `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution.
        - `'threshold'`: squashes to zero all coefficients less than alpha from
          the projection ``dictionary * X'``.

    transform_n_nonzero_coefs : int, default=None
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and
        `algorithm='omp'`. If `None`, then
        `transform_n_nonzero_coefs=int(n_features / 10)`.

    transform_alpha : float, default=None
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `None`, defaults to `alpha`.

        .. versionchanged:: 1.2
            When None, default value changed from 1.0 to `alpha`.

    verbose : bool or int, default=False
        To control the verbosity of the procedure.

    split_sign : bool, default=False
        Whether to split the sparse feature vector into the concatenation of
        its negative part and its positive part. This can improve the
        performance of downstream classifiers.

    random_state : int, RandomState instance or None, default=None
        Used for initializing the dictionary when ``dict_init`` is not
        specified, randomly shuffling the data when ``shuffle`` is set to
        ``True``, and updating the dictionary. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    transform_max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `'lasso_lars'`.

        .. versionadded:: 0.22

    callback : callable, default=None
        A callable that gets invoked at the end of each iteration.

        .. versionadded:: 1.1

    tol : float, default=1e-3
        Control early stopping based on the norm of the differences in the
        dictionary between 2 steps.

        To disable early stopping based on changes in the dictionary, set
        `tol` to 0.0.

        .. versionadded:: 1.1

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini batches
        that does not yield an improvement on the smoothed cost function.

        To disable convergence detection based on cost function, set
        `max_no_improvement` to None.

        .. versionadded:: 1.1

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Components extracted from the data.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations over the full dataset.

    n_steps_ : int
        Number of mini-batches processed.

        .. versionadded:: 1.1

    See Also
    --------
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.
    SparseCoder : Find a sparse representation of data from a fixed,
        precomputed dictionary.
    SparsePCA : Sparse Principal Components Analysis.

    References
    ----------

    J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
    for sparse coding (https://www.di.ens.fr/sierra/pdfs/icml09.pdf)

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> from sklearn.decomposition import MiniBatchDictionaryLearning
    >>> X, dictionary, code = make_sparse_coded_signal(
    ...     n_samples=30, n_components=15, n_features=20, n_nonzero_coefs=10,
    ...     random_state=42)
    >>> dict_learner = MiniBatchDictionaryLearning(
    ...     n_components=15, batch_size=3, transform_algorithm='lasso_lars',
    ...     transform_alpha=0.1, max_iter=20, random_state=42)
    >>> X_transformed = dict_learner.fit_transform(X)

    We can check the level of sparsity of `X_transformed`:

    >>> np.mean(X_transformed == 0) > 0.5
    True

    We can compare the average squared euclidean norm of the reconstruction
    error of the sparse coded signal relative to the squared euclidean norm of
    the original signal:

    >>> X_hat = X_transformed @ dict_learner.components_
    >>> np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1))
    0.052...
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left'), None], 'alpha': [Interval(Real, 0, None, closed='left')], 'max_iter': [Interval(Integral, 0, None, closed='left'), Hidden(None)], 'fit_algorithm': [StrOptions({'cd', 'lars'})], 'n_jobs': [None, Integral], 'batch_size': [Interval(Integral, 1, None, closed='left')], 'shuffle': ['boolean'], 'dict_init': [None, np.ndarray], 'transform_algorithm': [StrOptions({'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'})], 'transform_n_nonzero_coefs': [Interval(Integral, 1, None, closed='left'), None], 'transform_alpha': [Interval(Real, 0, None, closed='left'), None], 'verbose': ['verbose'], 'split_sign': ['boolean'], 'random_state': ['random_state'], 'positive_code': ['boolean'], 'positive_dict': ['boolean'], 'transform_max_iter': [Interval(Integral, 0, None, closed='left')], 'callback': [None, callable], 'tol': [Interval(Real, 0, None, closed='left')], 'max_no_improvement': [Interval(Integral, 0, None, closed='left'), None]}

    def __init__(self, n_components=None, *, alpha=1, max_iter=1000, fit_algorithm='lars', n_jobs=None, batch_size=256, shuffle=True, dict_init=None, transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, verbose=False, split_sign=False, random_state=None, positive_code=False, positive_dict=False, transform_max_iter=1000, callback=None, tol=0.001, max_no_improvement=10):
        super().__init__(transform_algorithm, transform_n_nonzero_coefs, transform_alpha, split_sign, n_jobs, positive_code, transform_max_iter)
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_algorithm = fit_algorithm
        self.dict_init = dict_init
        self.verbose = verbose
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.split_sign = split_sign
        self.random_state = random_state
        self.positive_dict = positive_dict
        self.callback = callback
        self.max_no_improvement = max_no_improvement
        self.tol = tol

    def _check_params(self, X):
        self._n_components = self.n_components
        if self._n_components is None:
            self._n_components = X.shape[1]
        _check_positive_coding(self.fit_algorithm, self.positive_code)
        self._fit_algorithm = 'lasso_' + self.fit_algorithm
        self._batch_size = min(self.batch_size, X.shape[0])

    def _initialize_dict(self, X, random_state):
        """Initialization of the dictionary."""
        if self.dict_init is not None:
            dictionary = self.dict_init
        else:
            _, S, dictionary = randomized_svd(X, self._n_components, random_state=random_state)
            dictionary = S[:, np.newaxis] * dictionary
        if self._n_components <= len(dictionary):
            dictionary = dictionary[:self._n_components, :]
        else:
            dictionary = np.concatenate((dictionary, np.zeros((self._n_components - len(dictionary), dictionary.shape[1]), dtype=dictionary.dtype)))
        dictionary = check_array(dictionary, order='F', dtype=X.dtype, copy=False)
        dictionary = np.require(dictionary, requirements='W')
        return dictionary

    def _update_inner_stats(self, X, code, batch_size, step):
        """Update the inner stats inplace."""
        if step < batch_size - 1:
            theta = (step + 1) * batch_size
        else:
            theta = batch_size ** 2 + step + 1 - batch_size
        beta = (theta + 1 - batch_size) / (theta + 1)
        self._A *= beta
        self._A += code.T @ code / batch_size
        self._B *= beta
        self._B += X.T @ code / batch_size

    def _minibatch_step(self, X, dictionary, random_state, step):
        """Perform the update on the dictionary for one minibatch."""
        batch_size = X.shape[0]
        code = _sparse_encode(X, dictionary, algorithm=self._fit_algorithm, alpha=self.alpha, n_jobs=self.n_jobs, positive=self.positive_code, max_iter=self.transform_max_iter, verbose=self.verbose)
        batch_cost = (0.5 * ((X - code @ dictionary) ** 2).sum() + self.alpha * np.sum(np.abs(code))) / batch_size
        self._update_inner_stats(X, code, batch_size, step)
        _update_dict(dictionary, X, code, self._A, self._B, verbose=self.verbose, random_state=random_state, positive=self.positive_dict)
        return batch_cost

    def _check_convergence(self, X, batch_cost, new_dict, old_dict, n_samples, step, n_steps):
        """Helper function to encapsulate the early stopping logic.

        Early stopping is based on two factors:
        - A small change of the dictionary between two minibatch updates. This is
          controlled by the tol parameter.
        - No more improvement on a smoothed estimate of the objective function for a
          a certain number of consecutive minibatch updates. This is controlled by
          the max_no_improvement parameter.
        """
        batch_size = X.shape[0]
        step = step + 1
        if step <= min(100, n_samples / batch_size):
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
        dict_diff = linalg.norm(new_dict - old_dict) / self._n_components
        if self.tol > 0 and dict_diff <= self.tol:
            if self.verbose:
                print(f'Converged (small dictionary change) at step {step}/{n_steps}')
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
    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X, dtype=[np.float64, np.float32], order='C', copy=False)
        self._check_params(X)
        self._random_state = check_random_state(self.random_state)
        dictionary = self._initialize_dict(X, self._random_state)
        old_dict = dictionary.copy()
        if self.shuffle:
            X_train = X.copy()
            self._random_state.shuffle(X_train)
        else:
            X_train = X
        n_samples, n_features = X_train.shape
        if self.verbose:
            print('[dict_learning]')
        self._A = np.zeros((self._n_components, self._n_components), dtype=X_train.dtype)
        self._B = np.zeros((n_features, self._n_components), dtype=X_train.dtype)
        if self.max_iter is None:
            warn('`max_iter=None` is deprecated in version 1.4 and will be removed in version 1.6. Use the default value (i.e. `1_000`) instead.', FutureWarning)
            max_iter = 1000
        else:
            max_iter = self.max_iter
        self._ewa_cost = None
        self._ewa_cost_min = None
        self._no_improvement = 0
        batches = gen_batches(n_samples, self._batch_size)
        batches = itertools.cycle(batches)
        n_steps_per_iter = int(np.ceil(n_samples / self._batch_size))
        n_steps = max_iter * n_steps_per_iter
        i = -1
        for i, batch in zip(range(n_steps), batches):
            X_batch = X_train[batch]
            batch_cost = self._minibatch_step(X_batch, dictionary, self._random_state, i)
            if self._check_convergence(X_batch, batch_cost, dictionary, old_dict, n_samples, i, n_steps):
                break
            if self.callback is not None:
                self.callback(locals())
            old_dict[:] = dictionary
        self.n_steps_ = i + 1
        self.n_iter_ = np.ceil(self.n_steps_ / n_steps_per_iter)
        self.components_ = dictionary
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        """Update the model using the data in X as a mini-batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        has_components = hasattr(self, 'components_')
        X = self._validate_data(X, dtype=[np.float64, np.float32], order='C', reset=not has_components)
        if not has_components:
            self._check_params(X)
            self._random_state = check_random_state(self.random_state)
            dictionary = self._initialize_dict(X, self._random_state)
            self.n_steps_ = 0
            self._A = np.zeros((self._n_components, self._n_components), dtype=X.dtype)
            self._B = np.zeros((X.shape[1], self._n_components), dtype=X.dtype)
        else:
            dictionary = self.components_
        self._minibatch_step(X, dictionary, self._random_state, self.n_steps_)
        self.components_ = dictionary
        self.n_steps_ += 1
        return self

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.components_.shape[0]

    def _more_tags(self):
        return {'preserves_dtype': [np.float64, np.float32]}