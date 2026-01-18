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
@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS + CSC_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS)
@pytest.mark.parametrize('sample_size', [None, 'half'])
def test_silhouette(sparse_container, sample_size):
    dataset = datasets.load_iris()
    X, y = (dataset.data, dataset.target)
    if sparse_container is not None:
        X = sparse_container(X)
    sample_size = int(X.shape[0] / 2) if sample_size == 'half' else sample_size
    D = pairwise_distances(X, metric='euclidean')
    score_precomputed = silhouette_score(D, y, metric='precomputed', sample_size=sample_size, random_state=0)
    score_euclidean = silhouette_score(X, y, metric='euclidean', sample_size=sample_size, random_state=0)
    assert score_precomputed > 0
    assert score_euclidean > 0
    assert score_precomputed == pytest.approx(score_euclidean)