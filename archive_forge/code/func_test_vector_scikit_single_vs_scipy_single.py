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
def test_vector_scikit_single_vs_scipy_single(global_random_seed):
    n_samples, n_features, n_clusters = (10, 5, 3)
    rng = np.random.RandomState(global_random_seed)
    X = 0.1 * rng.normal(size=(n_samples, n_features))
    X -= 4.0 * np.arange(n_samples)[:, np.newaxis]
    X -= X.mean(axis=1)[:, np.newaxis]
    out = hierarchy.linkage(X, method='single')
    children_scipy = out[:, :2].astype(int)
    children, _, n_leaves, _ = _TREE_BUILDERS['single'](X)
    children.sort(axis=1)
    assert_array_equal(children, children_scipy, 'linkage tree differs from scipy impl for single linkage.')
    cut = _hc_cut(n_clusters, children, n_leaves)
    cut_scipy = _hc_cut(n_clusters, children_scipy, n_leaves)
    assess_same_labelling(cut, cut_scipy)