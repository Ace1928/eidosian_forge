import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._unsupervised import _silhouette_reduce
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS)
def test_silhouette_reduce(sparse_container):
    """Check for non-CSR input to private method `_silhouette_reduce`."""
    X = np.array([[0.2, 0.1, 0.1, 0.2, 0.1, 1.6, 0.2, 0.1]], dtype=np.float32).T
    pdist_dense = pairwise_distances(X)
    pdist_sparse = sparse_container(pdist_dense)
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    label_freqs = np.bincount(y)
    with pytest.raises(TypeError, match='Expected CSR matrix. Please pass sparse matrix in CSR format.'):
        _silhouette_reduce(pdist_sparse, start=0, labels=y, label_freqs=label_freqs)