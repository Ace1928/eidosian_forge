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
def test_partial_dependence_kind_list(pyplot, clf_diabetes, diabetes):
    """Check that we can provide a list of strings to kind parameter."""
    matplotlib = pytest.importorskip('matplotlib')
    disp = PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, features=[0, 2, (1, 2)], grid_resolution=20, kind=['both', 'both', 'average'])
    for idx in [0, 1]:
        assert all([isinstance(line, matplotlib.lines.Line2D) for line in disp.lines_[0, idx].ravel()])
        assert disp.contours_[0, idx] is None
    assert disp.contours_[0, 2] is not None
    assert all([line is None for line in disp.lines_[0, 2].ravel()])