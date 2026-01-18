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
def test_radius_neighbors_regressor(n_samples=40, n_features=3, n_test_pts=10, radius=0.5, random_state=0):
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = np.sqrt((X ** 2).sum(1))
    y /= y.max()
    y_target = y[:n_test_pts]
    weight_func = _weight_func
    for algorithm in ALGORITHMS:
        for weights in ['uniform', 'distance', weight_func]:
            neigh = neighbors.RadiusNeighborsRegressor(radius=radius, weights=weights, algorithm=algorithm)
            neigh.fit(X, y)
            epsilon = 1e-05 * (2 * rng.rand(1, n_features) - 1)
            y_pred = neigh.predict(X[:n_test_pts] + epsilon)
            assert np.all(abs(y_pred - y_target) < radius / 2)
    for weights in ['uniform', 'distance']:
        neigh = neighbors.RadiusNeighborsRegressor(radius=radius, weights=weights, algorithm='auto')
        neigh.fit(X, y)
        X_test_nan = np.full((1, n_features), -1.0)
        empty_warning_msg = 'One or more samples have no neighbors within specified radius; predicting NaN.'
        with pytest.warns(UserWarning, match=re.escape(empty_warning_msg)):
            pred = neigh.predict(X_test_nan)
        assert np.all(np.isnan(pred))