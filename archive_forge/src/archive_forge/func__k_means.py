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
def _k_means(self, data, n_clusters):
    if self.mini_batch:
        model = MiniBatchKMeans(n_clusters, init=self.init, n_init=self.n_init, random_state=self.random_state)
    else:
        model = KMeans(n_clusters, init=self.init, n_init=self.n_init, random_state=self.random_state)
    model.fit(data)
    centroid = model.cluster_centers_
    labels = model.labels_
    return (centroid, labels)