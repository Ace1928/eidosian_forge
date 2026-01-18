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
def test_ward_agglomeration(global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    mask = np.ones([10, 10], dtype=bool)
    X = rng.randn(50, 100)
    connectivity = grid_to_graph(*mask.shape)
    agglo = FeatureAgglomeration(n_clusters=5, connectivity=connectivity)
    agglo.fit(X)
    assert np.size(np.unique(agglo.labels_)) == 5
    X_red = agglo.transform(X)
    assert X_red.shape[1] == 5
    X_full = agglo.inverse_transform(X_red)
    assert np.unique(X_full[0]).size == 5
    assert_array_almost_equal(agglo.transform(X_full), X_red)
    with pytest.raises(ValueError):
        agglo.fit(X[:0])