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
def test_ward_tree_children_order(global_random_seed):
    n, p = (10, 5)
    rng = np.random.RandomState(global_random_seed)
    connectivity = np.ones((n, n))
    for i in range(5):
        X = 0.1 * rng.normal(size=(n, p))
        X -= 4.0 * np.arange(n)[:, np.newaxis]
        X -= X.mean(axis=1)[:, np.newaxis]
        out_unstructured = ward_tree(X)
        out_structured = ward_tree(X, connectivity=connectivity)
        assert_array_equal(out_unstructured[0], out_structured[0])