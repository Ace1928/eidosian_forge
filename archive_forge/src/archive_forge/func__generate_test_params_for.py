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
def _generate_test_params_for(metric: str, n_features: int):
    """Return list of DistanceMetric kwargs for tests."""
    rng = np.random.RandomState(1)
    if metric == 'minkowski':
        minkowski_kwargs = [dict(p=1.5), dict(p=2), dict(p=3), dict(p=np.inf)]
        if sp_version >= parse_version('1.8.0.dev0'):
            minkowski_kwargs.append(dict(p=3, w=rng.rand(n_features)))
        return minkowski_kwargs
    if metric == 'seuclidean':
        return [dict(V=rng.rand(n_features))]
    if metric == 'mahalanobis':
        A = rng.rand(n_features, n_features)
        VI = A + A.T + 3 * np.eye(n_features)
        return [dict(VI=VI)]
    return [{}]