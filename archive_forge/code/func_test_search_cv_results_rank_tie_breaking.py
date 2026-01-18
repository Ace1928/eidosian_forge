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
def test_search_cv_results_rank_tie_breaking():
    X, y = make_blobs(n_samples=50, random_state=42)
    param_grid = {'C': [1, 1.001, 0.001]}
    grid_search = GridSearchCV(SVC(), param_grid=param_grid, return_train_score=True)
    random_search = RandomizedSearchCV(SVC(), n_iter=3, param_distributions=param_grid, return_train_score=True)
    for search in (grid_search, random_search):
        search.fit(X, y)
        cv_results = search.cv_results_
        assert_almost_equal(cv_results['mean_test_score'][0], cv_results['mean_test_score'][1])
        assert_almost_equal(cv_results['mean_train_score'][0], cv_results['mean_train_score'][1])
        assert not np.allclose(cv_results['mean_test_score'][1], cv_results['mean_test_score'][2])
        assert not np.allclose(cv_results['mean_train_score'][1], cv_results['mean_train_score'][2])
        assert_almost_equal(search.cv_results_['rank_test_score'], [1, 1, 3])