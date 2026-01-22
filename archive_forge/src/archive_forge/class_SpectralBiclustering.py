from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
from scipy.linalg import norm
from scipy.sparse import dia_matrix, issparse
from scipy.sparse.linalg import eigsh, svds
from ..base import BaseEstimator, BiclusterMixin, _fit_context
from ..utils import check_random_state, check_scalar
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import make_nonnegative, randomized_svd, safe_sparse_dot
from ..utils.validation import assert_all_finite
from ._kmeans import KMeans, MiniBatchKMeans
class SpectralBiclustering(BaseSpectral):
    """Spectral biclustering (Kluger, 2003).

    Partitions rows and columns under the assumption that the data has
    an underlying checkerboard structure. For instance, if there are
    two row partitions and three column partitions, each row will
    belong to three biclusters, and each column will belong to two
    biclusters. The outer product of the corresponding row and column
    label vectors gives this checkerboard structure.

    Read more in the :ref:`User Guide <spectral_biclustering>`.

    Parameters
    ----------
    n_clusters : int or tuple (n_row_clusters, n_column_clusters), default=3
        The number of row and column clusters in the checkerboard
        structure.

    method : {'bistochastic', 'scale', 'log'}, default='bistochastic'
        Method of normalizing and converting singular vectors into
        biclusters. May be one of 'scale', 'bistochastic', or 'log'.
        The authors recommend using 'log'. If the data is sparse,
        however, log normalization will not work, which is why the
        default is 'bistochastic'.

        .. warning::
           if `method='log'`, the data must not be sparse.

    n_components : int, default=6
        Number of singular vectors to check.

    n_best : int, default=3
        Number of best singular vectors to which to project the data
        for clustering.

    svd_method : {'randomized', 'arpack'}, default='randomized'
        Selects the algorithm for finding singular vectors. May be
        'randomized' or 'arpack'. If 'randomized', uses
        :func:`~sklearn.utils.extmath.randomized_svd`, which may be faster
        for large matrices. If 'arpack', uses
        `scipy.sparse.linalg.svds`, which is more accurate, but
        possibly slower in some cases.

    n_svd_vecs : int, default=None
        Number of vectors to use in calculating the SVD. Corresponds
        to `ncv` when `svd_method=arpack` and `n_oversamples` when
        `svd_method` is 'randomized`.

    mini_batch : bool, default=False
        Whether to use mini-batch k-means, which is faster but may get
        different results.

    init : {'k-means++', 'random'} or ndarray of shape (n_clusters, n_features),             default='k-means++'
        Method for initialization of k-means algorithm; defaults to
        'k-means++'.

    n_init : int, default=10
        Number of random initializations that are tried with the
        k-means algorithm.

        If mini-batch k-means is used, the best initialization is
        chosen and the algorithm runs once. Otherwise, the algorithm
        is run for each initialization and the best solution chosen.

    random_state : int, RandomState instance, default=None
        Used for randomizing the singular value decomposition and the k-means
        initialization. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    rows_ : array-like of shape (n_row_clusters, n_rows)
        Results of the clustering. `rows[i, r]` is True if
        cluster `i` contains row `r`. Available only after calling ``fit``.

    columns_ : array-like of shape (n_column_clusters, n_columns)
        Results of the clustering, like `rows`.

    row_labels_ : array-like of shape (n_rows,)
        Row partition labels.

    column_labels_ : array-like of shape (n_cols,)
        Column partition labels.

    biclusters_ : tuple of two ndarrays
        The tuple contains the `rows_` and `columns_` arrays.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SpectralCoclustering : Spectral Co-Clustering algorithm (Dhillon, 2001).

    References
    ----------

    * :doi:`Kluger, Yuval, et. al., 2003. Spectral biclustering of microarray
      data: coclustering genes and conditions.
      <10.1101/gr.648603>`

    Examples
    --------
    >>> from sklearn.cluster import SpectralBiclustering
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = SpectralBiclustering(n_clusters=2, random_state=0).fit(X)
    >>> clustering.row_labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> clustering.column_labels_
    array([1, 0], dtype=int32)
    >>> clustering
    SpectralBiclustering(n_clusters=2, random_state=0)
    """
    _parameter_constraints: dict = {**BaseSpectral._parameter_constraints, 'n_clusters': [Interval(Integral, 1, None, closed='left'), tuple], 'method': [StrOptions({'bistochastic', 'scale', 'log'})], 'n_components': [Interval(Integral, 1, None, closed='left')], 'n_best': [Interval(Integral, 1, None, closed='left')]}

    def __init__(self, n_clusters=3, *, method='bistochastic', n_components=6, n_best=3, svd_method='randomized', n_svd_vecs=None, mini_batch=False, init='k-means++', n_init=10, random_state=None):
        super().__init__(n_clusters, svd_method, n_svd_vecs, mini_batch, init, n_init, random_state)
        self.method = method
        self.n_components = n_components
        self.n_best = n_best

    def _check_parameters(self, n_samples):
        if isinstance(self.n_clusters, Integral):
            if self.n_clusters > n_samples:
                raise ValueError(f'n_clusters should be <= n_samples={n_samples}. Got {self.n_clusters} instead.')
        else:
            try:
                n_row_clusters, n_column_clusters = self.n_clusters
                check_scalar(n_row_clusters, 'n_row_clusters', target_type=Integral, min_val=1, max_val=n_samples)
                check_scalar(n_column_clusters, 'n_column_clusters', target_type=Integral, min_val=1, max_val=n_samples)
            except (ValueError, TypeError) as e:
                raise ValueError(f'Incorrect parameter n_clusters has value: {self.n_clusters}. It should either be a single integer or an iterable with two integers: (n_row_clusters, n_column_clusters) And the values are should be in the range: (1, n_samples)') from e
        if self.n_best > self.n_components:
            raise ValueError(f'n_best={self.n_best} must be <= n_components={self.n_components}.')

    def _fit(self, X):
        n_sv = self.n_components
        if self.method == 'bistochastic':
            normalized_data = _bistochastic_normalize(X)
            n_sv += 1
        elif self.method == 'scale':
            normalized_data, _, _ = _scale_normalize(X)
            n_sv += 1
        elif self.method == 'log':
            normalized_data = _log_normalize(X)
        n_discard = 0 if self.method == 'log' else 1
        u, v = self._svd(normalized_data, n_sv, n_discard)
        ut = u.T
        vt = v.T
        try:
            n_row_clusters, n_col_clusters = self.n_clusters
        except TypeError:
            n_row_clusters = n_col_clusters = self.n_clusters
        best_ut = self._fit_best_piecewise(ut, self.n_best, n_row_clusters)
        best_vt = self._fit_best_piecewise(vt, self.n_best, n_col_clusters)
        self.row_labels_ = self._project_and_cluster(X, best_vt.T, n_row_clusters)
        self.column_labels_ = self._project_and_cluster(X.T, best_ut.T, n_col_clusters)
        self.rows_ = np.vstack([self.row_labels_ == label for label in range(n_row_clusters) for _ in range(n_col_clusters)])
        self.columns_ = np.vstack([self.column_labels_ == label for _ in range(n_row_clusters) for label in range(n_col_clusters)])

    def _fit_best_piecewise(self, vectors, n_best, n_clusters):
        """Find the ``n_best`` vectors that are best approximated by piecewise
        constant vectors.

        The piecewise vectors are found by k-means; the best is chosen
        according to Euclidean distance.

        """

        def make_piecewise(v):
            centroid, labels = self._k_means(v.reshape(-1, 1), n_clusters)
            return centroid[labels].ravel()
        piecewise_vectors = np.apply_along_axis(make_piecewise, axis=1, arr=vectors)
        dists = np.apply_along_axis(norm, axis=1, arr=vectors - piecewise_vectors)
        result = vectors[np.argsort(dists)[:n_best]]
        return result

    def _project_and_cluster(self, data, vectors, n_clusters):
        """Project ``data`` to ``vectors`` and cluster the result."""
        projected = safe_sparse_dot(data, vectors)
        _, labels = self._k_means(projected, n_clusters)
        return labels