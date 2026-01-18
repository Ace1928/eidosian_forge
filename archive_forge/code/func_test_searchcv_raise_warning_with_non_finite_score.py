import pickle
import re
import sys
from collections.abc import Iterable, Sized
from functools import partial
from io import StringIO
from itertools import chain, product
from types import GeneratorType
import numpy as np
import pytest
from scipy.stats import bernoulli, expon, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import (
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.neighbors import KernelDensity, KNeighborsClassifier, LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
@pytest.mark.parametrize('return_train_score', [False, True])
@pytest.mark.parametrize('SearchCV, specialized_params', [(GridSearchCV, {'param_grid': {'max_depth': [2, 3, 5, 8]}}), (RandomizedSearchCV, {'param_distributions': {'max_depth': [2, 3, 5, 8]}, 'n_iter': 4})])
def test_searchcv_raise_warning_with_non_finite_score(SearchCV, specialized_params, return_train_score):
    X, y = make_classification(n_classes=2, random_state=0)

    class FailingScorer:
        """Scorer that will fail for some split but not all."""

        def __init__(self):
            self.n_counts = 0

        def __call__(self, estimator, X, y):
            self.n_counts += 1
            if self.n_counts % 5 == 0:
                return np.nan
            return 1
    grid = SearchCV(DecisionTreeClassifier(), scoring=FailingScorer(), cv=3, return_train_score=return_train_score, **specialized_params)
    with pytest.warns(UserWarning) as warn_msg:
        grid.fit(X, y)
    set_with_warning = ['test', 'train'] if return_train_score else ['test']
    assert len(warn_msg) == len(set_with_warning)
    for msg, dataset in zip(warn_msg, set_with_warning):
        assert f'One or more of the {dataset} scores are non-finite' in str(msg.message)
    last_rank = grid.cv_results_['rank_test_score'].max()
    non_finite_mask = np.isnan(grid.cv_results_['mean_test_score'])
    assert_array_equal(grid.cv_results_['rank_test_score'][non_finite_mask], last_rank)
    assert np.all(grid.cv_results_['rank_test_score'][~non_finite_mask] < last_rank)