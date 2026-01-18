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
def test_stratified_kfold_no_shuffle():
    X, y = (np.ones(4), [1, 1, 0, 0])
    splits = StratifiedKFold(2).split(X, y)
    train, test = next(splits)
    assert_array_equal(test, [0, 2])
    assert_array_equal(train, [1, 3])
    train, test = next(splits)
    assert_array_equal(test, [1, 3])
    assert_array_equal(train, [0, 2])
    X, y = (np.ones(7), [1, 1, 1, 0, 0, 0, 0])
    splits = StratifiedKFold(2).split(X, y)
    train, test = next(splits)
    assert_array_equal(test, [0, 1, 3, 4])
    assert_array_equal(train, [2, 5, 6])
    train, test = next(splits)
    assert_array_equal(test, [2, 5, 6])
    assert_array_equal(train, [0, 1, 3, 4])
    assert 5 == StratifiedKFold(5).get_n_splits(X, y)
    X = np.ones(7)
    y1 = ['1', '1', '1', '0', '0', '0', '0']
    y2 = [1, 1, 1, 0, 0, 0, 0]
    np.testing.assert_equal(list(StratifiedKFold(2).split(X, y1)), list(StratifiedKFold(2).split(X, y2)))
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    X = np.ones_like(y)
    np.testing.assert_equal(list(StratifiedKFold(3).split(X, y)), list(KFold(3).split(X, y)))