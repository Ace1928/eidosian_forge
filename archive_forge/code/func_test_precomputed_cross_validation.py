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
def test_precomputed_cross_validation():
    rng = np.random.RandomState(0)
    X = rng.rand(20, 2)
    D = pairwise_distances(X, metric='euclidean')
    y = rng.randint(3, size=20)
    for Est in (neighbors.KNeighborsClassifier, neighbors.RadiusNeighborsClassifier, neighbors.KNeighborsRegressor, neighbors.RadiusNeighborsRegressor):
        metric_score = cross_val_score(Est(), X, y)
        precomp_score = cross_val_score(Est(metric='precomputed'), D, y)
        assert_array_equal(metric_score, precomp_score)