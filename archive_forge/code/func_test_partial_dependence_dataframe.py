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
@pytest.mark.parametrize('estimator', [LogisticRegression(max_iter=1000, random_state=0), GradientBoostingClassifier(random_state=0, n_estimators=5)], ids=['estimator-brute', 'estimator-recursion'])
@pytest.mark.parametrize('preprocessor', [None, make_column_transformer((StandardScaler(), [iris.feature_names[i] for i in (0, 2)]), (RobustScaler(), [iris.feature_names[i] for i in (1, 3)])), make_column_transformer((StandardScaler(), [iris.feature_names[i] for i in (0, 2)]), remainder='passthrough')], ids=['None', 'column-transformer', 'column-transformer-passthrough'])
@pytest.mark.parametrize('features', [[0, 2], [iris.feature_names[i] for i in (0, 2)]], ids=['features-integer', 'features-string'])
def test_partial_dependence_dataframe(estimator, preprocessor, features):
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame(scale(iris.data), columns=iris.feature_names)
    pipe = make_pipeline(preprocessor, estimator)
    pipe.fit(df, iris.target)
    pdp_pipe = partial_dependence(pipe, df, features=features, grid_resolution=10, kind='average')
    if preprocessor is not None:
        X_proc = clone(preprocessor).fit_transform(df)
        features_clf = [0, 1]
    else:
        X_proc = df
        features_clf = [0, 2]
    clf = clone(estimator).fit(X_proc, iris.target)
    pdp_clf = partial_dependence(clf, X_proc, features=features_clf, method='brute', grid_resolution=10, kind='average')
    assert_allclose(pdp_pipe['average'], pdp_clf['average'])
    if preprocessor is not None:
        scaler = preprocessor.named_transformers_['standardscaler']
        assert_allclose(pdp_pipe['grid_values'][1], pdp_clf['grid_values'][1] * scaler.scale_[1] + scaler.mean_[1])
    else:
        assert_allclose(pdp_pipe['grid_values'][1], pdp_clf['grid_values'][1])