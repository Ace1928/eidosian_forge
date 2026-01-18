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
@pytest.mark.parametrize('algorithm', ALGORITHMS)
@pytest.mark.parametrize('weights', WEIGHTS)
def test_radius_neighbors_classifier(global_dtype, algorithm, weights, n_samples=40, n_features=5, n_test_pts=10, radius=0.5, random_state=0):
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features).astype(global_dtype, copy=False) - 1
    y = ((X ** 2).sum(axis=1) < radius).astype(int)
    y_str = y.astype(str)
    neigh = neighbors.RadiusNeighborsClassifier(radius=radius, weights=weights, algorithm=algorithm)
    neigh.fit(X, y)
    epsilon = 1e-05 * (2 * rng.rand(1, n_features) - 1)
    y_pred = neigh.predict(X[:n_test_pts] + epsilon)
    assert_array_equal(y_pred, y[:n_test_pts])
    neigh.fit(X, y_str)
    y_pred = neigh.predict(X[:n_test_pts] + epsilon)
    assert_array_equal(y_pred, y_str[:n_test_pts])