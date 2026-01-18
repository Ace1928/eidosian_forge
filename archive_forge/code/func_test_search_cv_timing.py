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
@ignore_warnings()
def test_search_cv_timing():
    svc = LinearSVC(dual='auto', random_state=0)
    X = [[1], [2], [3], [4]]
    y = [0, 1, 1, 0]
    gs = GridSearchCV(svc, {'C': [0, 1]}, cv=2, error_score=0)
    rs = RandomizedSearchCV(svc, {'C': [0, 1]}, cv=2, error_score=0, n_iter=2)
    for search in (gs, rs):
        search.fit(X, y)
        for key in ['mean_fit_time', 'std_fit_time']:
            assert np.all(search.cv_results_[key] >= 0)
            assert np.all(search.cv_results_[key] < 1)
        for key in ['mean_score_time', 'std_score_time']:
            assert search.cv_results_[key][1] >= 0
            assert search.cv_results_[key][0] == 0.0
            assert np.all(search.cv_results_[key] < 1)
        assert hasattr(search, 'refit_time_')
        assert isinstance(search.refit_time_, float)
        assert search.refit_time_ >= 0