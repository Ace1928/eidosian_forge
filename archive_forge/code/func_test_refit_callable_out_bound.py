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
@pytest.mark.parametrize('out_bound_value', [-1, 2])
@pytest.mark.parametrize('search_cv', [RandomizedSearchCV, GridSearchCV])
def test_refit_callable_out_bound(out_bound_value, search_cv):
    """
    Test implementation catches the errors when 'best_index_' returns an
    out of bound result.
    """

    def refit_callable_out_bound(cv_results):
        """
        A dummy function tests when returned 'best_index_' is out of bounds.
        """
        return out_bound_value
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    clf = search_cv(LinearSVC(dual='auto', random_state=42), {'C': [0.1, 1]}, scoring='precision', refit=refit_callable_out_bound)
    with pytest.raises(IndexError, match='best_index_ index out of range'):
        clf.fit(X, y)