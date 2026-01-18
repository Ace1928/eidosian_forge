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
@pytest.mark.parametrize('kind, centered, subsample, shape', [('average', False, None, (1, 3)), ('individual', False, None, (1, 3, 50)), ('both', False, None, (1, 3, 51)), ('individual', False, 20, (1, 3, 20)), ('both', False, 20, (1, 3, 21)), ('individual', False, 0.5, (1, 3, 25)), ('both', False, 0.5, (1, 3, 26)), ('average', True, None, (1, 3)), ('individual', True, None, (1, 3, 50)), ('both', True, None, (1, 3, 51)), ('individual', True, 20, (1, 3, 20)), ('both', True, 20, (1, 3, 21))])
def test_plot_partial_dependence_kind(pyplot, kind, centered, subsample, shape, clf_diabetes, diabetes):
    disp = PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, [0, 1, 2], kind=kind, centered=centered, subsample=subsample)
    assert disp.axes_.shape == (1, 3)
    assert disp.lines_.shape == shape
    assert disp.contours_.shape == (1, 3)
    assert disp.contours_[0, 0] is None
    assert disp.contours_[0, 1] is None
    assert disp.contours_[0, 2] is None
    if centered:
        assert all([ln._y[0] == 0.0 for ln in disp.lines_.ravel() if ln is not None])
    else:
        assert all([ln._y[0] != 0.0 for ln in disp.lines_.ravel() if ln is not None])