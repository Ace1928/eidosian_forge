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
def test_stratified_shuffle_split_multilabel():
    for y in [np.array([[0, 1], [1, 0], [1, 0], [0, 1]]), np.array([[0, 1], [1, 1], [1, 1], [0, 1]])]:
        X = np.ones_like(y)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        train, test = next(sss.split(X=X, y=y))
        y_train = y[train]
        y_test = y[test]
        assert_array_equal(np.intersect1d(train, test), [])
        assert_array_equal(np.union1d(train, test), np.arange(len(y)))
        expected_ratio = np.mean(y[:, 0])
        assert expected_ratio == np.mean(y_train[:, 0])
        assert expected_ratio == np.mean(y_test[:, 0])