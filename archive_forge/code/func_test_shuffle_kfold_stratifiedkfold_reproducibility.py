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
@pytest.mark.parametrize('kfold', [KFold, StratifiedKFold, StratifiedGroupKFold])
def test_shuffle_kfold_stratifiedkfold_reproducibility(kfold):
    X = np.ones(15)
    y = [0] * 7 + [1] * 8
    groups_1 = np.arange(len(y))
    X2 = np.ones(16)
    y2 = [0] * 8 + [1] * 8
    groups_2 = np.arange(len(y2))
    kf = kfold(3, shuffle=True, random_state=0)
    np.testing.assert_equal(list(kf.split(X, y, groups_1)), list(kf.split(X, y, groups_1)))
    kf = kfold(3, shuffle=True, random_state=np.random.RandomState(0))
    for data in zip((X, X2), (y, y2), (groups_1, groups_2)):
        for (_, test_a), (_, test_b) in zip(kf.split(*data), kf.split(*data)):
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(test_a, test_b)