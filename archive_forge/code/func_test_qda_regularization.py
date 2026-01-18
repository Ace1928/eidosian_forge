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
@pytest.mark.xfail(_IS_WASM, reason='no floating point exceptions, see https://github.com/numpy/numpy/pull/21895#issuecomment-1311525881')
def test_qda_regularization():
    collinear_msg = 'Variables are collinear'
    clf = QuadraticDiscriminantAnalysis()
    with pytest.warns(UserWarning, match=collinear_msg):
        y_pred = clf.fit(X2, y6)
    with pytest.warns(RuntimeWarning, match='divide by zero'):
        y_pred = clf.predict(X2)
    assert np.any(y_pred != y6)
    clf = QuadraticDiscriminantAnalysis(reg_param=0.01)
    with pytest.warns(UserWarning, match=collinear_msg):
        clf.fit(X2, y6)
    y_pred = clf.predict(X2)
    assert_array_equal(y_pred, y6)
    clf = QuadraticDiscriminantAnalysis(reg_param=0.1)
    with pytest.warns(UserWarning, match=collinear_msg):
        clf.fit(X5, y5)
    y_pred5 = clf.predict(X5)
    assert_array_equal(y_pred5, y5)