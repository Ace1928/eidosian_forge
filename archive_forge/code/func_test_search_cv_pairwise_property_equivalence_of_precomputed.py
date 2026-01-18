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
def test_search_cv_pairwise_property_equivalence_of_precomputed():
    """
    Test implementation of BaseSearchCV has the pairwise tag
    which matches the pairwise tag of its estimator.
    This test ensures the equivalence of 'precomputed'.

    Non-regression test for issue #13920.
    """
    n_samples = 50
    n_splits = 2
    X, y = make_classification(n_samples=n_samples, random_state=0)
    grid_params = {'n_neighbors': [10]}
    clf = KNeighborsClassifier()
    cv = GridSearchCV(clf, grid_params, cv=n_splits)
    cv.fit(X, y)
    preds_original = cv.predict(X)
    X_precomputed = euclidean_distances(X)
    clf = KNeighborsClassifier(metric='precomputed')
    cv = GridSearchCV(clf, grid_params, cv=n_splits)
    cv.fit(X_precomputed, y)
    preds_precomputed = cv.predict(X_precomputed)
    attr_message = 'GridSearchCV not identical with precomputed metric'
    assert (preds_original == preds_precomputed).all(), attr_message