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
@pytest.mark.parametrize('param_distributions, expected_n_candidates', [({'a': [1, 2]}, 2), ({'a': randint(1, 3)}, 10)])
def test_random_search_discrete_distributions(param_distributions, expected_n_candidates):
    n_samples = 1024
    X, y = make_classification(n_samples=n_samples, random_state=0)
    base_estimator = FastClassifier()
    sh = HalvingRandomSearchCV(base_estimator, param_distributions, n_candidates=10)
    sh.fit(X, y)
    assert sh.n_candidates_[0] == expected_n_candidates