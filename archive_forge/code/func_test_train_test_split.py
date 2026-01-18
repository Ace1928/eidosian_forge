import re
import warnings
from itertools import combinations, combinations_with_replacement, permutations
import numpy as np
import pytest
from scipy import stats
from scipy.sparse import issparse
from scipy.special import comb
from sklearn import config_context
from sklearn.datasets import load_digits, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
from sklearn.model_selection._split import (
from sklearn.svm import SVC
from sklearn.tests.metadata_routing_common import assert_request_is_empty
from sklearn.utils._array_api import (
from sklearn.utils._array_api import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
def test_train_test_split(coo_container):
    X = np.arange(100).reshape((10, 10))
    X_s = coo_container(X)
    y = np.arange(10)
    split = train_test_split(X, y, test_size=None, train_size=0.5)
    X_train, X_test, y_train, y_test = split
    assert len(y_test) == len(y_train)
    assert_array_equal(X_train[:, 0], y_train * 10)
    assert_array_equal(X_test[:, 0], y_test * 10)
    split = train_test_split(X, X_s, y.tolist())
    X_train, X_test, X_s_train, X_s_test, y_train, y_test = split
    assert isinstance(y_train, list)
    assert isinstance(y_test, list)
    X_4d = np.arange(10 * 5 * 3 * 2).reshape(10, 5, 3, 2)
    y_3d = np.arange(10 * 7 * 11).reshape(10, 7, 11)
    split = train_test_split(X_4d, y_3d)
    assert split[0].shape == (7, 5, 3, 2)
    assert split[1].shape == (3, 5, 3, 2)
    assert split[2].shape == (7, 7, 11)
    assert split[3].shape == (3, 7, 11)
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    for test_size, exp_test_size in zip([2, 4, 0.25, 0.5, 0.75], [2, 4, 2, 4, 6]):
        train, test = train_test_split(y, test_size=test_size, stratify=y, random_state=0)
        assert len(test) == exp_test_size
        assert len(test) + len(train) == len(y)
        assert np.sum(train == 1) == np.sum(train == 2)
    y = np.arange(10)
    for test_size in [2, 0.2]:
        train, test = train_test_split(y, shuffle=False, test_size=test_size)
        assert_array_equal(test, [8, 9])
        assert_array_equal(train, [0, 1, 2, 3, 4, 5, 6, 7])