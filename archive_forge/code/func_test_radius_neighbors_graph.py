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
def test_radius_neighbors_graph():
    X = np.array([[0, 1], [1.01, 1.0], [2, 0]])
    A = neighbors.radius_neighbors_graph(X, 1.5, mode='connectivity', include_self=True)
    assert_array_equal(A.toarray(), [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    A = neighbors.radius_neighbors_graph(X, 1.5, mode='distance')
    assert_allclose(A.toarray(), [[0.0, 1.01, 0.0], [1.01, 0.0, 1.40716026], [0.0, 1.40716026, 0.0]])