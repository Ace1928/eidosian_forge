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
@pytest.mark.parametrize('kind, expected_shape', [('average', (1, 2)), ('individual', (1, 2, 20)), ('both', (1, 2, 21))])
def test_plot_partial_dependence_subsampling(pyplot, clf_diabetes, diabetes, kind, expected_shape):
    matplotlib = pytest.importorskip('matplotlib')
    grid_resolution = 25
    feature_names = diabetes.feature_names
    disp1 = PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, ['age', 'bmi'], kind=kind, grid_resolution=grid_resolution, feature_names=feature_names, subsample=20, random_state=0)
    assert disp1.lines_.shape == expected_shape
    assert all([isinstance(line, matplotlib.lines.Line2D) for line in disp1.lines_.ravel()])