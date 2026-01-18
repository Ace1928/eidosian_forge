import itertools
import shutil
from functools import partial
from tempfile import mkdtemp
import numpy as np
import pytest
from scipy.cluster import hierarchy
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree
from sklearn.cluster._agglomerative import (
from sklearn.cluster._hierarchical_fast import (
from sklearn.datasets import make_circles, make_moons
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.metrics import DistanceMetric
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import (
from sklearn.metrics.tests.test_dist_metrics import METRICS_DEFAULT_PARAMS
from sklearn.neighbors import kneighbors_graph
from sklearn.utils._fast_dict import IntFloatDict
from sklearn.utils._testing import (
from sklearn.utils.fixes import LIL_CONTAINERS
def test_precomputed_connectivity_metric_with_2_connected_components():
    """Check that connecting components works when connectivity and
    affinity are both precomputed and the number of connected components is
    greater than 1. Non-regression test for #16151.
    """
    connectivity_matrix = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
    assert connected_components(connectivity_matrix)[0] == 2
    rng = np.random.RandomState(0)
    X = rng.randn(5, 10)
    X_dist = pairwise_distances(X)
    clusterer_precomputed = AgglomerativeClustering(metric='precomputed', connectivity=connectivity_matrix, linkage='complete')
    msg = 'Completing it to avoid stopping the tree early'
    with pytest.warns(UserWarning, match=msg):
        clusterer_precomputed.fit(X_dist)
    clusterer = AgglomerativeClustering(connectivity=connectivity_matrix, linkage='complete')
    with pytest.warns(UserWarning, match=msg):
        clusterer.fit(X)
    assert_array_equal(clusterer.labels_, clusterer_precomputed.labels_)
    assert_array_equal(clusterer.children_, clusterer_precomputed.children_)