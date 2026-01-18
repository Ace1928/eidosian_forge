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
@pytest.mark.parametrize('SearchCV', [HalvingGridSearchCV, HalvingRandomSearchCV])
def test_select_best_index(SearchCV):
    """Check the selection strategy of the halving search."""
    results = {'iter': np.array([0, 0, 0, 0, 1, 1, 2, 2, 2]), 'mean_test_score': np.array([4, 3, 5, 1, 11, 10, 5, 6, 9]), 'params': np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])}
    best_index = SearchCV._select_best_index(None, None, results)
    assert best_index == 8