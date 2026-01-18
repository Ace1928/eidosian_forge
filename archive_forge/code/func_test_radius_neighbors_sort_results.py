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
@pytest.mark.parametrize(['algorithm', 'metric'], list(product(('kd_tree', 'ball_tree', 'brute'), ('euclidean', *DISTANCE_METRIC_OBJS))) + [('brute', 'euclidean'), ('brute', 'precomputed')])
def test_radius_neighbors_sort_results(algorithm, metric):
    metric = _parse_metric(metric, np.float64)
    if isinstance(metric, DistanceMetric):
        pytest.skip('Metrics of type `DistanceMetric` are not yet supported for radius-neighbor estimators.')
    n_samples = 10
    rng = np.random.RandomState(42)
    X = rng.random_sample((n_samples, 4))
    if metric == 'precomputed':
        X = neighbors.radius_neighbors_graph(X, radius=np.inf, mode='distance')
    model = neighbors.NearestNeighbors(algorithm=algorithm, metric=metric)
    model.fit(X)
    distances, indices = model.radius_neighbors(X=X, radius=np.inf, sort_results=True)
    for ii in range(n_samples):
        assert_array_equal(distances[ii], np.sort(distances[ii]))
    if metric != 'precomputed':
        with pytest.raises(ValueError, match='return_distance must be True'):
            model.radius_neighbors(X=X, radius=np.inf, sort_results=True, return_distance=False)
    graph = model.radius_neighbors_graph(X=X, radius=np.inf, mode='distance', sort_results=True)
    assert _is_sorted_by_data(graph)