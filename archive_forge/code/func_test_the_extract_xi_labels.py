import warnings
import numpy as np
import pytest
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.cluster._optics import _extend_region, _extract_xi_labels
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize(('ordering', 'clusters', 'expected'), [[[0, 1, 2, 3], [[0, 1], [2, 3]], [0, 0, 1, 1]], [[0, 1, 2, 3], [[0, 1], [3, 3]], [0, 0, -1, 1]], [[0, 1, 2, 3], [[0, 1], [3, 3], [0, 3]], [0, 0, -1, 1]], [[3, 1, 2, 0], [[0, 1], [3, 3], [0, 3]], [1, 0, -1, 0]]])
def test_the_extract_xi_labels(ordering, clusters, expected):
    labels = _extract_xi_labels(ordering, clusters)
    assert_array_equal(labels, expected)