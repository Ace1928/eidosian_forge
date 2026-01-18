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
def test_partial_dependence_pipeline():
    iris = load_iris()
    scaler = StandardScaler()
    clf = DummyClassifier(random_state=42)
    pipe = make_pipeline(scaler, clf)
    clf.fit(scaler.fit_transform(iris.data), iris.target)
    pipe.fit(iris.data, iris.target)
    features = 0
    pdp_pipe = partial_dependence(pipe, iris.data, features=[features], grid_resolution=10, kind='average')
    pdp_clf = partial_dependence(clf, scaler.transform(iris.data), features=[features], grid_resolution=10, kind='average')
    assert_allclose(pdp_pipe['average'], pdp_clf['average'])
    assert_allclose(pdp_pipe['grid_values'][0], pdp_clf['grid_values'][0] * scaler.scale_[features] + scaler.mean_[features])