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
@pytest.mark.parametrize('metric', neighbors.VALID_METRICS['brute'] + DISTANCE_METRIC_OBJS)
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_valid_brute_metric_for_auto_algorithm(global_dtype, metric, csr_container, n_samples=20, n_features=12):
    metric = _parse_metric(metric, global_dtype)
    X = rng.rand(n_samples, n_features).astype(global_dtype, copy=False)
    Xcsr = csr_container(X)
    metric_params_list = _generate_test_params_for(metric, n_features)
    if metric == 'precomputed':
        X_precomputed = rng.random_sample((10, 4))
        Y_precomputed = rng.random_sample((3, 4))
        DXX = metrics.pairwise_distances(X_precomputed, metric='euclidean')
        DYX = metrics.pairwise_distances(Y_precomputed, X_precomputed, metric='euclidean')
        nb_p = neighbors.NearestNeighbors(n_neighbors=3, metric='precomputed')
        nb_p.fit(DXX)
        nb_p.kneighbors(DYX)
    else:
        for metric_params in metric_params_list:
            nn = neighbors.NearestNeighbors(n_neighbors=3, algorithm='auto', metric=metric, metric_params=metric_params)
            if metric == 'haversine':
                feature_sl = slice(None, 2)
                X = np.ascontiguousarray(X[:, feature_sl])
            nn.fit(X)
            nn.kneighbors(X)
            if metric in VALID_METRICS_SPARSE['brute']:
                nn = neighbors.NearestNeighbors(n_neighbors=3, algorithm='auto', metric=metric).fit(Xcsr)
                nn.kneighbors(Xcsr)