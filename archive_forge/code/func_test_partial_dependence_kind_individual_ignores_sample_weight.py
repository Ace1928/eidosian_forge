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
@pytest.mark.parametrize('Estimator, data', [(LinearRegression, multioutput_regression_data), (LogisticRegression, binary_classification_data)])
def test_partial_dependence_kind_individual_ignores_sample_weight(Estimator, data):
    """Check that `sample_weight` does not have any effect on reported ICE."""
    est = Estimator()
    (X, y), n_targets = data
    sample_weight = np.arange(X.shape[0])
    est.fit(X, y)
    pdp_nsw = partial_dependence(est, X=X, features=[1, 2], kind='individual')
    pdp_sw = partial_dependence(est, X=X, features=[1, 2], kind='individual', sample_weight=sample_weight)
    assert_allclose(pdp_nsw['individual'], pdp_sw['individual'])
    assert_allclose(pdp_nsw['grid_values'], pdp_sw['grid_values'])