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
def test_shuffle_kfold():
    kf = KFold(3)
    kf2 = KFold(3, shuffle=True, random_state=0)
    kf3 = KFold(3, shuffle=True, random_state=1)
    X = np.ones(300)
    all_folds = np.zeros(300)
    for (tr1, te1), (tr2, te2), (tr3, te3) in zip(kf.split(X), kf2.split(X), kf3.split(X)):
        for tr_a, tr_b in combinations((tr1, tr2, tr3), 2):
            assert len(np.intersect1d(tr_a, tr_b)) != len(tr1)
        all_folds[te2] = 1
    assert sum(all_folds) == 300