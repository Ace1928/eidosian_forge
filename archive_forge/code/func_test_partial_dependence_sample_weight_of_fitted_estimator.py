import warnings
import numpy as np
import pytest
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_regressor
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.inspection import partial_dependence
from sklearn.inspection._partial_dependence import (
from sklearn.linear_model import LinearRegression, LogisticRegression, MultiTaskLasso
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree.tests.test_tree import assert_is_subtree
from sklearn.utils import _IS_32BIT
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.validation import check_random_state
def test_partial_dependence_sample_weight_of_fitted_estimator():
    N = 1000
    rng = np.random.RandomState(123456)
    mask = rng.randint(2, size=N, dtype=bool)
    x = rng.rand(N)
    y = x.copy()
    y[~mask] = -y[~mask]
    X = np.c_[mask, x]
    sample_weight = np.ones(N)
    sample_weight[mask] = 1000.0
    clf = GradientBoostingRegressor(n_estimators=10, random_state=1)
    clf.fit(X, y, sample_weight=sample_weight)
    pdp = partial_dependence(clf, X, features=[1], kind='average')
    assert np.corrcoef(pdp['average'], pdp['grid_values'])[0, 1] > 0.99