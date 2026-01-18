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
def test_same_knn_parallel(algorithm):
    X, y = datasets.make_classification(n_samples=30, n_features=5, n_redundant=0, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = neighbors.KNeighborsClassifier(n_neighbors=3, algorithm=algorithm)
    clf.fit(X_train, y_train)
    y = clf.predict(X_test)
    dist, ind = clf.kneighbors(X_test)
    graph = clf.kneighbors_graph(X_test, mode='distance').toarray()
    clf.set_params(n_jobs=3)
    clf.fit(X_train, y_train)
    y_parallel = clf.predict(X_test)
    dist_parallel, ind_parallel = clf.kneighbors(X_test)
    graph_parallel = clf.kneighbors_graph(X_test, mode='distance').toarray()
    assert_array_equal(y, y_parallel)
    assert_allclose(dist, dist_parallel)
    assert_array_equal(ind, ind_parallel)
    assert_allclose(graph, graph_parallel)