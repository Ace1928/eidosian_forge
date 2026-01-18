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
@pytest.mark.parametrize('estimator', [LinearRegression(), LogisticRegression(), RandomForestRegressor(), GradientBoostingClassifier()])
@pytest.mark.parametrize('non_null_weight_idx', [0, 1, -1])
def test_partial_dependence_non_null_weight_idx(estimator, non_null_weight_idx):
    """Check that if we pass a `sample_weight` of zeros with only one index with
    sample weight equals one, then the average `partial_dependence` with this
    `sample_weight` is equal to the individual `partial_dependence` of the
    corresponding index.
    """
    X, y = (iris.data, iris.target)
    preprocessor = make_column_transformer((StandardScaler(), [0, 2]), (RobustScaler(), [1, 3]))
    pipe = make_pipeline(preprocessor, estimator).fit(X, y)
    sample_weight = np.zeros_like(y)
    sample_weight[non_null_weight_idx] = 1
    pdp_sw = partial_dependence(pipe, X, [2, 3], kind='average', sample_weight=sample_weight, grid_resolution=10)
    pdp_ind = partial_dependence(pipe, X, [2, 3], kind='individual', grid_resolution=10)
    output_dim = 1 if is_regressor(pipe) else len(np.unique(y))
    for i in range(output_dim):
        assert_allclose(pdp_ind['individual'][i][non_null_weight_idx], pdp_sw['average'][i])