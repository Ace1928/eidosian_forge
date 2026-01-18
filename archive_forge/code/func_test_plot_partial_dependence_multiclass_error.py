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
@pytest.mark.parametrize('params, err_msg', [({'target': 4, 'features': [0]}, 'target not in est.classes_, got 4'), ({'target': None, 'features': [0]}, 'target must be specified for multi-class'), ({'target': 1, 'features': [4.5]}, 'Each entry in features must be either an int,')])
def test_plot_partial_dependence_multiclass_error(pyplot, params, err_msg):
    iris = load_iris()
    clf = GradientBoostingClassifier(n_estimators=10, random_state=1)
    clf.fit(iris.data, iris.target)
    with pytest.raises(ValueError, match=err_msg):
        PartialDependenceDisplay.from_estimator(clf, iris.data, **params)