import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats.mstats import mquantiles
from sklearn.compose import make_column_transformer
from sklearn.datasets import (
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._testing import _convert_container
@pytest.mark.filterwarnings('ignore:A Bunch will be returned')
@pytest.mark.parametrize('data, params, err_msg', [(multioutput_regression_data, {'target': None, 'features': [0]}, 'target must be specified for multi-output'), (multioutput_regression_data, {'target': -1, 'features': [0]}, 'target must be in \\[0, n_tasks\\]'), (multioutput_regression_data, {'target': 100, 'features': [0]}, 'target must be in \\[0, n_tasks\\]'), (dummy_classification_data, {'features': ['foobar'], 'feature_names': None}, "Feature 'foobar' not in feature_names"), (dummy_classification_data, {'features': ['foobar'], 'feature_names': ['abcd', 'def']}, "Feature 'foobar' not in feature_names"), (dummy_classification_data, {'features': [(1, 2, 3)]}, 'Each entry in features must be either an int, '), (dummy_classification_data, {'features': [1, {}]}, 'Each entry in features must be either an int, '), (dummy_classification_data, {'features': [tuple()]}, 'Each entry in features must be either an int, '), (dummy_classification_data, {'features': [123], 'feature_names': ['blahblah']}, 'All entries of features must be less than '), (dummy_classification_data, {'features': [0, 1, 2], 'feature_names': ['a', 'b', 'a']}, 'feature_names should not contain duplicates'), (dummy_classification_data, {'features': [1, 2], 'kind': ['both']}, 'When `kind` is provided as a list of strings, it should contain'), (dummy_classification_data, {'features': [1], 'subsample': -1}, 'When an integer, subsample=-1 should be positive.'), (dummy_classification_data, {'features': [1], 'subsample': 1.2}, 'When a floating-point, subsample=1.2 should be in the \\(0, 1\\) range'), (dummy_classification_data, {'features': [1, 2], 'categorical_features': [1.0, 2.0]}, 'Expected `categorical_features` to be an array-like of boolean,'), (dummy_classification_data, {'features': [(1, 2)], 'categorical_features': [2]}, 'Two-way partial dependence plots are not supported for pairs'), (dummy_classification_data, {'features': [1], 'categorical_features': [1], 'kind': 'individual'}, 'It is not possible to display individual effects')])
def test_plot_partial_dependence_error(pyplot, data, params, err_msg):
    X, y = data
    estimator = LinearRegression().fit(X, y)
    with pytest.raises(ValueError, match=err_msg):
        PartialDependenceDisplay.from_estimator(estimator, X, **params)