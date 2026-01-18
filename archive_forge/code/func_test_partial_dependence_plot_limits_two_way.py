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
@pytest.mark.parametrize('centered', [True, False])
def test_partial_dependence_plot_limits_two_way(pyplot, clf_diabetes, diabetes, centered):
    """Check that the PD limit on the plots are properly set on two-way plots."""
    disp = PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, features=[(0, 1)], kind='average', grid_resolution=25, feature_names=diabetes.feature_names)
    range_pd = np.array([-1, 1], dtype=np.float64)
    for pd in disp.pd_results:
        pd['average'][...] = range_pd[1]
        pd['average'][0, 0] = range_pd[0]
    disp.plot(centered=centered)
    contours = disp.contours_[0, 0]
    levels = range_pd - range_pd[0] if centered else range_pd
    padding = 0.05 * (levels[1] - levels[0])
    levels[0] -= padding
    levels[1] += padding
    expect_levels = np.linspace(*levels, num=8)
    assert_allclose(contours.levels, expect_levels)