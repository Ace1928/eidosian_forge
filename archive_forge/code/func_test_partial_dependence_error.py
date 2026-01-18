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
@pytest.mark.filterwarnings('ignore:A Bunch will be returned')
@pytest.mark.parametrize('estimator, params, err_msg', [(KMeans(random_state=0, n_init='auto'), {'features': [0]}, "'estimator' must be a fitted regressor or classifier"), (LinearRegression(), {'features': [0], 'response_method': 'predict_proba'}, 'The response_method parameter is ignored for regressors'), (GradientBoostingClassifier(random_state=0), {'features': [0], 'response_method': 'predict_proba', 'method': 'recursion'}, "'recursion' method, the response_method must be 'decision_function'"), (GradientBoostingClassifier(random_state=0), {'features': [0], 'response_method': 'predict_proba', 'method': 'auto'}, "'recursion' method, the response_method must be 'decision_function'"), (LinearRegression(), {'features': [0], 'method': 'recursion', 'kind': 'individual'}, "The 'recursion' method only applies when 'kind' is set to 'average'"), (LinearRegression(), {'features': [0], 'method': 'recursion', 'kind': 'both'}, "The 'recursion' method only applies when 'kind' is set to 'average'"), (LinearRegression(), {'features': [0], 'method': 'recursion'}, "Only the following estimators support the 'recursion' method:")])
def test_partial_dependence_error(estimator, params, err_msg):
    X, y = make_classification(random_state=0)
    estimator.fit(X, y)
    with pytest.raises(ValueError, match=err_msg):
        partial_dependence(estimator, X, **params)