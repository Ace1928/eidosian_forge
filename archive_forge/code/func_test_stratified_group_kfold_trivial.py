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
def test_stratified_group_kfold_trivial():
    sgkf = StratifiedGroupKFold(n_splits=3)
    y = np.array([1] * 6 + [0] * 12)
    X = np.ones_like(y).reshape(-1, 1)
    groups = np.asarray((1, 2, 3, 4, 5, 6, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6))
    distr = np.bincount(y) / len(y)
    test_sizes = []
    for train, test in sgkf.split(X, y, groups):
        assert np.intersect1d(groups[train], groups[test]).size == 0
        assert_allclose(np.bincount(y[train]) / len(train), distr, atol=0.02)
        assert_allclose(np.bincount(y[test]) / len(test), distr, atol=0.02)
        test_sizes.append(len(test))
    assert np.ptp(test_sizes) <= 1