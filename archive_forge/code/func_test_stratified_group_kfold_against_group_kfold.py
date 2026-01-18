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
@pytest.mark.parametrize('cls_distr', [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.8, 0.2)])
@pytest.mark.parametrize('n_groups', [5, 30, 70])
def test_stratified_group_kfold_against_group_kfold(cls_distr, n_groups):
    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    gkf = GroupKFold(n_splits=n_splits)
    rng = np.random.RandomState(0)
    n_points = 1000
    y = rng.choice(2, size=n_points, p=cls_distr)
    X = np.ones_like(y).reshape(-1, 1)
    g = rng.choice(n_groups, n_points)
    sgkf_folds = sgkf.split(X, y, groups=g)
    gkf_folds = gkf.split(X, y, groups=g)
    sgkf_entr = 0
    gkf_entr = 0
    for (sgkf_train, sgkf_test), (_, gkf_test) in zip(sgkf_folds, gkf_folds):
        assert np.intersect1d(g[sgkf_train], g[sgkf_test]).size == 0
        sgkf_distr = np.bincount(y[sgkf_test]) / len(sgkf_test)
        gkf_distr = np.bincount(y[gkf_test]) / len(gkf_test)
        sgkf_entr += stats.entropy(sgkf_distr, qk=cls_distr)
        gkf_entr += stats.entropy(gkf_distr, qk=cls_distr)
    sgkf_entr /= n_splits
    gkf_entr /= n_splits
    assert sgkf_entr <= gkf_entr