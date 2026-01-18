import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_pair_confusion_matrix():
    n = 10
    N = n ** 2
    clustering1 = np.hstack([[i + 1] * n for i in range(n)])
    clustering2 = np.hstack([[i + 1] * (n + 1) for i in range(n)])[:N]
    expected = np.zeros(shape=(2, 2), dtype=np.int64)
    for i in range(len(clustering1)):
        for j in range(len(clustering2)):
            if i != j:
                same_cluster_1 = int(clustering1[i] == clustering1[j])
                same_cluster_2 = int(clustering2[i] == clustering2[j])
                expected[same_cluster_1, same_cluster_2] += 1
    assert_array_equal(pair_confusion_matrix(clustering1, clustering2), expected)