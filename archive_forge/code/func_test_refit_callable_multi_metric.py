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
def test_refit_callable_multi_metric():
    """
    Test refit=callable in multiple metric evaluation setting
    """

    def refit_callable(cv_results):
        """
        A dummy function tests `refit=callable` interface.
        Return the index of a model that has the least
        `mean_test_prec`.
        """
        assert 'mean_test_prec' in cv_results
        return cv_results['mean_test_prec'].argmin()
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    scoring = {'Accuracy': make_scorer(accuracy_score), 'prec': 'precision'}
    clf = GridSearchCV(LinearSVC(dual='auto', random_state=42), {'C': [0.01, 0.1, 1]}, scoring=scoring, refit=refit_callable)
    clf.fit(X, y)
    assert clf.best_index_ == 0
    assert not hasattr(clf, 'best_score_')