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
@pytest.mark.parametrize('max_resources, n_candidates, expected_n_candidates', [(512, 'exhaust', 128), (32, 'exhaust', 8), (32, 8, 8), (32, 7, 7), (32, 9, 9)])
def test_random_search(max_resources, n_candidates, expected_n_candidates):
    n_samples = 1024
    X, y = make_classification(n_samples=n_samples, random_state=0)
    param_grid = {'a': norm, 'b': norm}
    base_estimator = FastClassifier()
    sh = HalvingRandomSearchCV(base_estimator, param_grid, n_candidates=n_candidates, cv=2, max_resources=max_resources, factor=2, min_resources=4)
    sh.fit(X, y)
    assert sh.n_candidates_[0] == expected_n_candidates
    if n_candidates == 'exhaust':
        assert sh.n_resources_[-1] == max_resources