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
def _project_and_cluster(self, data, vectors, n_clusters):
    """Project ``data`` to ``vectors`` and cluster the result."""
    projected = safe_sparse_dot(data, vectors)
    _, labels = self._k_means(projected, n_clusters)
    return labels