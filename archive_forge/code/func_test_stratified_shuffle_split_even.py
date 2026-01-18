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
def test_stratified_shuffle_split_even():
    n_folds = 5
    n_splits = 1000

    def assert_counts_are_ok(idx_counts, p):
        threshold = 0.05 / n_splits
        bf = stats.binom(n_splits, p)
        for count in idx_counts:
            prob = bf.pmf(count)
            assert prob > threshold, 'An index is not drawn with chance corresponding to even draws'
    for n_samples in (6, 22):
        groups = np.array(n_samples // 2 * [0, 1])
        splits = StratifiedShuffleSplit(n_splits=n_splits, test_size=1.0 / n_folds, random_state=0)
        train_counts = [0] * n_samples
        test_counts = [0] * n_samples
        n_splits_actual = 0
        for train, test in splits.split(X=np.ones(n_samples), y=groups):
            n_splits_actual += 1
            for counter, ids in [(train_counts, train), (test_counts, test)]:
                for id in ids:
                    counter[id] += 1
        assert n_splits_actual == n_splits
        n_train, n_test = _validate_shuffle_split(n_samples, test_size=1.0 / n_folds, train_size=1.0 - 1.0 / n_folds)
        assert len(train) == n_train
        assert len(test) == n_test
        assert len(set(train).intersection(test)) == 0
        group_counts = np.unique(groups)
        assert splits.test_size == 1.0 / n_folds
        assert n_train + n_test == len(groups)
        assert len(group_counts) == 2
        ex_test_p = float(n_test) / n_samples
        ex_train_p = float(n_train) / n_samples
        assert_counts_are_ok(train_counts, ex_train_p)
        assert_counts_are_ok(test_counts, ex_test_p)