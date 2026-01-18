import numpy as np
import pytest
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf, ShrunkCovariance, ledoit_wolf
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import (
from sklearn.preprocessing import StandardScaler
from sklearn.utils import _IS_WASM, check_random_state
from sklearn.utils._testing import (
def test_qda():
    clf = QuadraticDiscriminantAnalysis()
    y_pred = clf.fit(X6, y6).predict(X6)
    assert_array_equal(y_pred, y6)
    y_pred1 = clf.fit(X7, y6).predict(X7)
    assert_array_equal(y_pred1, y6)
    y_proba_pred1 = clf.predict_proba(X7)
    assert_array_equal((y_proba_pred1[:, 1] > 0.5) + 1, y6)
    y_log_proba_pred1 = clf.predict_log_proba(X7)
    assert_array_almost_equal(np.exp(y_log_proba_pred1), y_proba_pred1, 8)
    y_pred3 = clf.fit(X6, y7).predict(X6)
    assert np.any(y_pred3 != y7)
    with pytest.raises(ValueError):
        clf.fit(X6, y4)