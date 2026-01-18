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
def test_regressor_predict_on_arraylikes():
    """Ensures that `predict` works for array-likes when `weights` is a callable.

    Non-regression test for #22687.
    """
    X = [[5, 1], [3, 1], [4, 3], [0, 3]]
    y = [2, 3, 5, 6]

    def _weights(dist):
        return np.ones_like(dist)
    est = KNeighborsRegressor(n_neighbors=1, algorithm='brute', weights=_weights)
    est.fit(X, y)
    assert_allclose(est.predict([[0, 2.5]]), [6])