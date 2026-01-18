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
@pytest.mark.parametrize('kind', ['individual', 'both', 'average', ['average', 'both'], ['individual', 'both']])
def test_partial_dependence_display_kind_centered_interaction(pyplot, kind, clf_diabetes, diabetes):
    """Check that we properly center ICE and PD when passing kind as a string and as a
    list."""
    disp = PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, [0, 1], kind=kind, centered=True, subsample=5)
    assert all([ln._y[0] == 0.0 for ln in disp.lines_.ravel() if ln is not None])