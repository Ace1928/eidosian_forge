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
def test_stratified_shuffle_split_iter():
    ys = [np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]), np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2] * 2), np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]), np.array([-1] * 800 + [1] * 50), np.concatenate([[i] * (100 + i) for i in range(11)]), [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3], ['1', '1', '1', '1', '2', '2', '2', '3', '3', '3', '3', '3']]
    for y in ys:
        sss = StratifiedShuffleSplit(6, test_size=0.33, random_state=0).split(np.ones(len(y)), y)
        y = np.asanyarray(y)
        test_size = np.ceil(0.33 * len(y))
        train_size = len(y) - test_size
        for train, test in sss:
            assert_array_equal(np.unique(y[train]), np.unique(y[test]))
            p_train = np.bincount(np.unique(y[train], return_inverse=True)[1]) / float(len(y[train]))
            p_test = np.bincount(np.unique(y[test], return_inverse=True)[1]) / float(len(y[test]))
            assert_array_almost_equal(p_train, p_test, 1)
            assert len(train) + len(test) == y.size
            assert len(train) == train_size
            assert len(test) == test_size
            assert_array_equal(np.intersect1d(train, test), [])