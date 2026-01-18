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
@pytest.mark.parametrize('n_samples, n_features, n_query_pts, n_neighbors', [(100, 100, 10, 100), (1000, 5, 100, 1)])
@pytest.mark.parametrize('query_is_train', [False, True])
@pytest.mark.parametrize('metric', COMMON_VALID_METRICS + DISTANCE_METRIC_OBJS)
def test_unsupervised_kneighbors(global_dtype, n_samples, n_features, n_query_pts, n_neighbors, query_is_train, metric):
    metric = _parse_metric(metric, global_dtype)
    local_rng = np.random.RandomState(0)
    X = local_rng.rand(n_samples, n_features).astype(global_dtype, copy=False)
    query = X if query_is_train else local_rng.rand(n_query_pts, n_features).astype(global_dtype, copy=False)
    results_nodist = []
    results = []
    for algorithm in ALGORITHMS:
        if isinstance(metric, DistanceMetric) and global_dtype == np.float32:
            if 'tree' in algorithm:
                pytest.skip('Neither KDTree nor BallTree support 32-bit distance metric objects.')
        neigh = neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        neigh.fit(X)
        results_nodist.append(neigh.kneighbors(query, return_distance=False))
        results.append(neigh.kneighbors(query, return_distance=True))
    for i in range(len(results) - 1):
        algorithm = ALGORITHMS[i]
        next_algorithm = ALGORITHMS[i + 1]
        indices_no_dist = results_nodist[i]
        distances, next_distances = (results[i][0], results[i + 1][0])
        indices, next_indices = (results[i][1], results[i + 1][1])
        assert_array_equal(indices_no_dist, indices, err_msg=f"The '{algorithm}' algorithm returns differentindices depending on 'return_distances'.")
        assert_array_equal(indices, next_indices, err_msg=f"The '{algorithm}' and '{next_algorithm}' algorithms return different indices.")
        assert_allclose(distances, next_distances, err_msg=f"The '{algorithm}' and '{next_algorithm}' algorithms return different distances.", atol=1e-06)