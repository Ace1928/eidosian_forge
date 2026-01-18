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
def test_train_test_split_empty_trainset():
    X, = [[1]]
    with pytest.raises(ValueError, match='With n_samples=1, test_size=0.99 and train_size=None, the resulting train set will be empty'):
        train_test_split(X, test_size=0.99)
    X = [[1], [1], [1]]
    with pytest.raises(ValueError, match='With n_samples=3, test_size=0.67 and train_size=None, the resulting train set will be empty'):
        train_test_split(X, test_size=0.67)