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
@pytest.mark.parametrize('SearchCV, param_search', [(GridSearchCV, 'param_grid'), (RandomizedSearchCV, 'param_distributions'), (HalvingGridSearchCV, 'param_grid')])
def test_score_rejects_params_with_no_routing_enabled(SearchCV, param_search):
    """*SearchCV should reject **params when metadata routing is not enabled
    since this is added only when routing is enabled."""
    X, y = make_classification(random_state=42)
    est = LinearSVC(dual='auto')
    param_grid_search = {param_search: {'C': [1]}}
    gs = SearchCV(est, cv=2, **param_grid_search).fit(X, y)
    with pytest.raises(ValueError, match='is only supported if'):
        gs.score(X, y, metadata=1)