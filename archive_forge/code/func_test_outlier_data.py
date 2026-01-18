import numpy as np
import pytest
from scipy import stats
from scipy.spatial import distance
from sklearn.cluster import HDBSCAN
from sklearn.cluster._hdbscan._tree import (
from sklearn.cluster._hdbscan.hdbscan import _OUTLIER_ENCODING
from sklearn.datasets import make_blobs
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics.pairwise import _VALID_METRICS, euclidean_distances
from sklearn.neighbors import BallTree, KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('outlier_type', _OUTLIER_ENCODING)
def test_outlier_data(outlier_type):
    """
    Tests if np.inf and np.nan data are each treated as special outliers.
    """
    outlier = {'infinite': np.inf, 'missing': np.nan}[outlier_type]
    prob_check = {'infinite': lambda x, y: x == y, 'missing': lambda x, y: np.isnan(x)}[outlier_type]
    label = _OUTLIER_ENCODING[outlier_type]['label']
    prob = _OUTLIER_ENCODING[outlier_type]['prob']
    X_outlier = X.copy()
    X_outlier[0] = [outlier, 1]
    X_outlier[5] = [outlier, outlier]
    model = HDBSCAN().fit(X_outlier)
    missing_labels_idx, = (model.labels_ == label).nonzero()
    assert_array_equal(missing_labels_idx, [0, 5])
    missing_probs_idx, = prob_check(model.probabilities_, prob).nonzero()
    assert_array_equal(missing_probs_idx, [0, 5])
    clean_indices = list(range(1, 5)) + list(range(6, 200))
    clean_model = HDBSCAN().fit(X_outlier[clean_indices])
    assert_array_equal(clean_model.labels_, model.labels_[clean_indices])