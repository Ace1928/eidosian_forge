from itertools import product
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import config_context, datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.utils._array_api import yield_namespace_device_dtype_combinations
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import (
def test_safe_split_with_precomputed_kernel():
    clf = SVC()
    clfp = SVC(kernel='precomputed')
    iris = datasets.load_iris()
    X, y = (iris.data, iris.target)
    K = np.dot(X, X.T)
    cv = ShuffleSplit(test_size=0.25, random_state=0)
    train, test = list(cv.split(X))[0]
    X_train, y_train = _safe_split(clf, X, y, train)
    K_train, y_train2 = _safe_split(clfp, K, y, train)
    assert_array_almost_equal(K_train, np.dot(X_train, X_train.T))
    assert_array_almost_equal(y_train, y_train2)
    X_test, y_test = _safe_split(clf, X, y, test, train)
    K_test, y_test2 = _safe_split(clfp, K, y, test, train)
    assert_array_almost_equal(K_test, np.dot(X_test, X_train.T))
    assert_array_almost_equal(y_test, y_test2)