from math import ceil
import numpy as np
import pytest
from scipy.stats import expon, norm, randint
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
from sklearn.model_selection._search_successive_halving import (
from sklearn.model_selection.tests.test_search import (
from sklearn.svm import SVC, LinearSVC
@pytest.mark.parametrize('fraction, subsample_test, expected_train_size, expected_test_size', [(0.5, True, 40, 10), (0.5, False, 40, 20), (0.2, True, 16, 4), (0.2, False, 16, 20)])
def test_subsample_splitter_shapes(fraction, subsample_test, expected_train_size, expected_test_size):
    n_samples = 100
    X, y = make_classification(n_samples)
    cv = _SubsampleMetaSplitter(base_cv=KFold(5), fraction=fraction, subsample_test=subsample_test, random_state=None)
    for train, test in cv.split(X, y):
        assert train.shape[0] == expected_train_size
        assert test.shape[0] == expected_test_size
        if subsample_test:
            assert train.shape[0] + test.shape[0] == int(n_samples * fraction)
        else:
            assert test.shape[0] == n_samples // cv.base_cv.get_n_splits()