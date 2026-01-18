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
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('k', [4, 5, 6, 7, 8, 9, 10])
@pytest.mark.parametrize('kfold', [StratifiedKFold, StratifiedGroupKFold])
def test_stratified_kfold_ratios(k, shuffle, kfold):
    n_samples = 1000
    X = np.ones(n_samples)
    y = np.array([4] * int(0.1 * n_samples) + [0] * int(0.89 * n_samples) + [1] * int(0.01 * n_samples))
    groups = np.arange(len(y))
    distr = np.bincount(y) / len(y)
    test_sizes = []
    random_state = None if not shuffle else 0
    skf = kfold(k, random_state=random_state, shuffle=shuffle)
    for train, test in skf.split(X, y, groups=groups):
        assert_allclose(np.bincount(y[train]) / len(train), distr, atol=0.02)
        assert_allclose(np.bincount(y[test]) / len(test), distr, atol=0.02)
        test_sizes.append(len(test))
    assert np.ptp(test_sizes) <= 1