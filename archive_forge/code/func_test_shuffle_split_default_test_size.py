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
@pytest.mark.parametrize('split_class', [ShuffleSplit, StratifiedShuffleSplit])
@pytest.mark.parametrize('train_size, exp_train, exp_test', [(None, 9, 1), (8, 8, 2), (0.8, 8, 2)])
def test_shuffle_split_default_test_size(split_class, train_size, exp_train, exp_test):
    X = np.ones(10)
    y = np.ones(10)
    X_train, X_test = next(split_class(train_size=train_size).split(X, y))
    assert len(X_train) == exp_train
    assert len(X_test) == exp_test