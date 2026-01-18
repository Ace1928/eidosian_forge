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
def test_partial_dependence_sample_weight_with_recursion():
    """Check that we raise an error when `sample_weight` is provided with
    `"recursion"` method.
    """
    est = RandomForestRegressor()
    (X, y), n_targets = regression_data
    sample_weight = np.ones_like(y)
    est.fit(X, y, sample_weight=sample_weight)
    with pytest.raises(ValueError, match="'recursion' method can only be applied when"):
        partial_dependence(est, X, features=[0], method='recursion', sample_weight=sample_weight)