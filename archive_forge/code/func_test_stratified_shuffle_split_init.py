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
@ignore_warnings
def test_stratified_shuffle_split_init():
    X = np.arange(7)
    y = np.asarray([0, 1, 1, 1, 2, 2, 2])
    with pytest.raises(ValueError):
        next(StratifiedShuffleSplit(3, test_size=0.2).split(X, y))
    with pytest.raises(ValueError):
        next(StratifiedShuffleSplit(3, test_size=2).split(X, y))
    with pytest.raises(ValueError):
        next(StratifiedShuffleSplit(3, test_size=3, train_size=2).split(X, y))
    X = np.arange(9)
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])
    with pytest.raises(ValueError):
        next(StratifiedShuffleSplit(train_size=2).split(X, y))
    with pytest.raises(ValueError):
        next(StratifiedShuffleSplit(test_size=2).split(X, y))