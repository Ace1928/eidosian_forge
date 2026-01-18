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
def test_warning_recursion_non_constant_init():
    gbc = GradientBoostingClassifier(init=DummyClassifier(), random_state=0)
    gbc.fit(X, y)
    with pytest.warns(UserWarning, match='Using recursion method with a non-constant init predictor'):
        partial_dependence(gbc, X, [0], method='recursion', kind='average')
    with pytest.warns(UserWarning, match='Using recursion method with a non-constant init predictor'):
        partial_dependence(gbc, X, [0], method='recursion', kind='average')