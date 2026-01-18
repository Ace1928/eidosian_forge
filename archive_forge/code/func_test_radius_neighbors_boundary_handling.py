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
def test_radius_neighbors_boundary_handling():
    """Test whether points lying on boundary are handled consistently

    Also ensures that even with only one query point, an object array
    is returned rather than a 2d array.
    """
    X = np.array([[1.5], [3.0], [3.01]])
    radius = 3.0
    for algorithm in ALGORITHMS:
        nbrs = neighbors.NearestNeighbors(radius=radius, algorithm=algorithm).fit(X)
        results = nbrs.radius_neighbors([[0.0]], return_distance=False)
        assert results.shape == (1,)
        assert results.dtype == object
        assert_array_equal(results[0], [0, 1])