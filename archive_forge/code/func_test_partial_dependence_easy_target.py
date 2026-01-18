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
@pytest.mark.parametrize('est', (LinearRegression(), GradientBoostingRegressor(random_state=0), HistGradientBoostingRegressor(random_state=0, min_samples_leaf=1, max_leaf_nodes=None, max_iter=1), DecisionTreeRegressor(random_state=0)))
@pytest.mark.parametrize('power', (1, 2))
def test_partial_dependence_easy_target(est, power):
    rng = np.random.RandomState(0)
    n_samples = 200
    target_variable = 2
    X = rng.normal(size=(n_samples, 5))
    y = X[:, target_variable] ** power
    est.fit(X, y)
    pdp = partial_dependence(est, features=[target_variable], X=X, grid_resolution=1000, kind='average')
    new_X = pdp['grid_values'][0].reshape(-1, 1)
    new_y = pdp['average'][0]
    new_X = PolynomialFeatures(degree=power).fit_transform(new_X)
    lr = LinearRegression().fit(new_X, new_y)
    r2 = r2_score(new_y, lr.predict(new_X))
    assert r2 > 0.99