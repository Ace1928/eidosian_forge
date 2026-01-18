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
@pytest.mark.parametrize('kind', ['individual', 'average', 'both'])
@pytest.mark.parametrize('centered', [True, False])
def test_partial_dependence_plot_limits_one_way(pyplot, clf_diabetes, diabetes, kind, centered):
    """Check that the PD limit on the plots are properly set on one-way plots."""
    disp = PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, features=(0, 1), kind=kind, grid_resolution=25, feature_names=diabetes.feature_names)
    range_pd = np.array([-1, 1], dtype=np.float64)
    for pd in disp.pd_results:
        if 'average' in pd:
            pd['average'][...] = range_pd[1]
            pd['average'][0, 0] = range_pd[0]
        if 'individual' in pd:
            pd['individual'][...] = range_pd[1]
            pd['individual'][0, 0, 0] = range_pd[0]
    disp.plot(centered=centered)
    y_lim = range_pd - range_pd[0] if centered else range_pd
    padding = 0.05 * (y_lim[1] - y_lim[0])
    y_lim[0] -= padding
    y_lim[1] += padding
    for ax in disp.axes_.ravel():
        assert_allclose(ax.get_ylim(), y_lim)