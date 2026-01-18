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
@pytest.mark.parametrize('CVSplitter', (ShuffleSplit, GroupShuffleSplit, StratifiedShuffleSplit))
def test_shuffle_split_empty_trainset(CVSplitter):
    cv = CVSplitter(test_size=0.99)
    X, y = ([[1]], [0])
    with pytest.raises(ValueError, match='With n_samples=1, test_size=0.99 and train_size=None, the resulting train set will be empty'):
        next(cv.split(X, y, groups=[1]))