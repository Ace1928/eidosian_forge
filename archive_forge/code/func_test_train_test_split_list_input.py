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
def test_train_test_split_list_input():
    X = np.ones(7)
    y1 = ['1'] * 4 + ['0'] * 3
    y2 = np.hstack((np.ones(4), np.zeros(3)))
    y3 = y2.tolist()
    for stratify in (True, False):
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, stratify=y1 if stratify else None, random_state=0)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, stratify=y2 if stratify else None, random_state=0)
        X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y3, stratify=y3 if stratify else None, random_state=0)
        np.testing.assert_equal(X_train1, X_train2)
        np.testing.assert_equal(y_train2, y_train3)
        np.testing.assert_equal(X_test1, X_test3)
        np.testing.assert_equal(y_test3, y_test2)