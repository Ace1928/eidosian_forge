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
@pytest.mark.filterwarnings('ignore:The total space of parameters 4 is')
@pytest.mark.parametrize('SearchCV', [GridSearchCV, RandomizedSearchCV])
@pytest.mark.parametrize('Predictor', [MinimalRegressor, MinimalClassifier])
def test_search_cv_using_minimal_compatible_estimator(SearchCV, Predictor):
    rng = np.random.RandomState(0)
    X, y = (rng.randn(25, 2), np.array([0] * 5 + [1] * 20))
    model = Pipeline([('transformer', MinimalTransformer()), ('predictor', Predictor())])
    params = {'transformer__param': [1, 10], 'predictor__parama': [1, 10]}
    search = SearchCV(model, params, error_score='raise')
    search.fit(X, y)
    assert search.best_params_.keys() == params.keys()
    y_pred = search.predict(X)
    if is_classifier(search):
        assert_array_equal(y_pred, 1)
        assert search.score(X, y) == pytest.approx(accuracy_score(y, y_pred))
    else:
        assert_allclose(y_pred, y.mean())
        assert search.score(X, y) == pytest.approx(r2_score(y, y_pred))