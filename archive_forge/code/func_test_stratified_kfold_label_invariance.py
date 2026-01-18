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
@pytest.mark.parametrize('k', [4, 6, 7])
@pytest.mark.parametrize('kfold', [StratifiedKFold, StratifiedGroupKFold])
def test_stratified_kfold_label_invariance(k, shuffle, kfold):
    n_samples = 100
    y = np.array([2] * int(0.1 * n_samples) + [0] * int(0.89 * n_samples) + [1] * int(0.01 * n_samples))
    X = np.ones(len(y))
    groups = np.arange(len(y))

    def get_splits(y):
        random_state = None if not shuffle else 0
        return [(list(train), list(test)) for train, test in kfold(k, random_state=random_state, shuffle=shuffle).split(X, y, groups=groups)]
    splits_base = get_splits(y)
    for perm in permutations([0, 1, 2]):
        y_perm = np.take(perm, y)
        splits_perm = get_splits(y_perm)
        assert splits_perm == splits_base