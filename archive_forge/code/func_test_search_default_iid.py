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
@pytest.mark.parametrize('SearchCV, specialized_params', [(GridSearchCV, {'param_grid': {'C': [1, 10]}}), (RandomizedSearchCV, {'param_distributions': {'C': [1, 10]}, 'n_iter': 2})])
def test_search_default_iid(SearchCV, specialized_params):
    X, y = make_blobs(centers=[[0, 0], [1, 0], [0, 1], [1, 1]], random_state=0, cluster_std=0.1, shuffle=False, n_samples=80)
    mask = np.ones(X.shape[0], dtype=bool)
    mask[np.where(y == 1)[0][::2]] = 0
    mask[np.where(y == 2)[0][::2]] = 0
    cv = [[mask, ~mask], [~mask, mask]]
    common_params = {'estimator': SVC(), 'cv': cv, 'return_train_score': True}
    search = SearchCV(**common_params, **specialized_params)
    search.fit(X, y)
    test_cv_scores = np.array([search.cv_results_['split%d_test_score' % s][0] for s in range(search.n_splits_)])
    test_mean = search.cv_results_['mean_test_score'][0]
    test_std = search.cv_results_['std_test_score'][0]
    train_cv_scores = np.array([search.cv_results_['split%d_train_score' % s][0] for s in range(search.n_splits_)])
    train_mean = search.cv_results_['mean_train_score'][0]
    train_std = search.cv_results_['std_train_score'][0]
    assert search.cv_results_['param_C'][0] == 1
    assert_allclose(test_cv_scores, [1, 1.0 / 3.0])
    assert_allclose(train_cv_scores, [1, 1])
    assert test_mean == pytest.approx(np.mean(test_cv_scores))
    assert test_std == pytest.approx(np.std(test_cv_scores))
    assert train_mean == pytest.approx(1)
    assert train_std == pytest.approx(0)