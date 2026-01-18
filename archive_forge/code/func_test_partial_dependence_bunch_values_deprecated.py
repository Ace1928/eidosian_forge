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
def test_partial_dependence_bunch_values_deprecated():
    """Test that deprecation warning is raised when values is accessed."""
    est = LogisticRegression()
    (X, y), _ = binary_classification_data
    est.fit(X, y)
    pdp_avg = partial_dependence(est, X=X, features=[1, 2], kind='average')
    msg = "Key: 'values', is deprecated in 1.3 and will be removed in 1.5. Please use 'grid_values' instead"
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        grid_values = pdp_avg['grid_values']
    with pytest.warns(FutureWarning, match=msg):
        values = pdp_avg['values']
    assert values is grid_values