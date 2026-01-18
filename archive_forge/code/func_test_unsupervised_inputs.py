import re
import warnings
from itertools import product
import joblib
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import (
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning, NotFittedError
from sklearn.metrics._dist_metrics import (
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import BOOL_METRICS
from sklearn.metrics.tests.test_pairwise_distances_reduction import (
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
from sklearn.neighbors._base import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('KNeighborsMixinSubclass', [neighbors.KNeighborsClassifier, neighbors.KNeighborsRegressor, neighbors.NearestNeighbors])
def test_unsupervised_inputs(global_dtype, KNeighborsMixinSubclass):
    X = rng.random_sample((10, 3)).astype(global_dtype, copy=False)
    y = rng.randint(3, size=10)
    nbrs_fid = neighbors.NearestNeighbors(n_neighbors=1)
    nbrs_fid.fit(X)
    dist1, ind1 = nbrs_fid.kneighbors(X)
    nbrs = KNeighborsMixinSubclass(n_neighbors=1)
    for data in (nbrs_fid, neighbors.BallTree(X), neighbors.KDTree(X)):
        nbrs.fit(data, y)
        dist2, ind2 = nbrs.kneighbors(X)
        assert_allclose(dist1, dist2)
        assert_array_equal(ind1, ind2)