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
class BaseSpectral(BiclusterMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for spectral biclustering."""
    _parameter_constraints: dict = {'svd_method': [StrOptions({'randomized', 'arpack'})], 'n_svd_vecs': [Interval(Integral, 0, None, closed='left'), None], 'mini_batch': ['boolean'], 'init': [StrOptions({'k-means++', 'random'}), np.ndarray], 'n_init': [Interval(Integral, 1, None, closed='left')], 'random_state': ['random_state']}

    @abstractmethod
    def __init__(self, n_clusters=3, svd_method='randomized', n_svd_vecs=None, mini_batch=False, init='k-means++', n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.svd_method = svd_method
        self.n_svd_vecs = n_svd_vecs
        self.mini_batch = mini_batch
        self.init = init
        self.n_init = n_init
        self.random_state = random_state

    @abstractmethod
    def _check_parameters(self, n_samples):
        """Validate parameters depending on the input data."""

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Create a biclustering for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            SpectralBiclustering instance.
        """
        X = self._validate_data(X, accept_sparse='csr', dtype=np.float64)
        self._check_parameters(X.shape[0])
        self._fit(X)
        return self

    def _svd(self, array, n_components, n_discard):
        """Returns first `n_components` left and right singular
        vectors u and v, discarding the first `n_discard`.
        """
        if self.svd_method == 'randomized':
            kwargs = {}
            if self.n_svd_vecs is not None:
                kwargs['n_oversamples'] = self.n_svd_vecs
            u, _, vt = randomized_svd(array, n_components, random_state=self.random_state, **kwargs)
        elif self.svd_method == 'arpack':
            u, _, vt = svds(array, k=n_components, ncv=self.n_svd_vecs)
            if np.any(np.isnan(vt)):
                A = safe_sparse_dot(array.T, array)
                random_state = check_random_state(self.random_state)
                v0 = random_state.uniform(-1, 1, A.shape[0])
                _, v = eigsh(A, ncv=self.n_svd_vecs, v0=v0)
                vt = v.T
            if np.any(np.isnan(u)):
                A = safe_sparse_dot(array, array.T)
                random_state = check_random_state(self.random_state)
                v0 = random_state.uniform(-1, 1, A.shape[0])
                _, u = eigsh(A, ncv=self.n_svd_vecs, v0=v0)
        assert_all_finite(u)
        assert_all_finite(vt)
        u = u[:, n_discard:]
        vt = vt[n_discard:]
        return (u, vt.T)

    def _k_means(self, data, n_clusters):
        if self.mini_batch:
            model = MiniBatchKMeans(n_clusters, init=self.init, n_init=self.n_init, random_state=self.random_state)
        else:
            model = KMeans(n_clusters, init=self.init, n_init=self.n_init, random_state=self.random_state)
        model.fit(data)
        centroid = model.cluster_centers_
        labels = model.labels_
        return (centroid, labels)

    def _more_tags(self):
        return {'_xfail_checks': {'check_estimators_dtypes': 'raises nan error', 'check_fit2d_1sample': '_scale_normalize fails', 'check_fit2d_1feature': 'raises apply_along_axis error', 'check_estimator_sparse_data': 'does not fail gracefully', 'check_methods_subset_invariance': 'empty array passed inside', 'check_dont_overwrite_parameters': 'empty array passed inside', 'check_fit2d_predict1d': 'empty array passed inside'}}