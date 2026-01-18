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
def test_plot_partial_dependence_feature_name_reuse(pyplot, clf_diabetes, diabetes):
    feature_names = diabetes.feature_names
    disp = PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, [0, 1], grid_resolution=10, feature_names=feature_names)
    PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, [0, 1], grid_resolution=10, ax=disp.axes_)
    for i, ax in enumerate(disp.axes_.ravel()):
        assert ax.get_xlabel() == feature_names[i]