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
@pytest.mark.parametrize('kind, lines', [('average', 1), ('individual', 50), ('both', 51)])
def test_plot_partial_dependence_passing_numpy_axes(pyplot, clf_diabetes, diabetes, kind, lines):
    grid_resolution = 25
    feature_names = diabetes.feature_names
    disp1 = PartialDependenceDisplay.from_estimator(clf_diabetes, diabetes.data, ['age', 'bmi'], kind=kind, grid_resolution=grid_resolution, feature_names=feature_names)
    assert disp1.axes_.shape == (1, 2)
    assert disp1.axes_[0, 0].get_ylabel() == 'Partial dependence'
    assert disp1.axes_[0, 1].get_ylabel() == ''
    assert len(disp1.axes_[0, 0].get_lines()) == lines
    assert len(disp1.axes_[0, 1].get_lines()) == lines
    lr = LinearRegression()
    lr.fit(diabetes.data, diabetes.target)
    disp2 = PartialDependenceDisplay.from_estimator(lr, diabetes.data, ['age', 'bmi'], kind=kind, grid_resolution=grid_resolution, feature_names=feature_names, ax=disp1.axes_)
    assert np.all(disp1.axes_ == disp2.axes_)
    assert len(disp2.axes_[0, 0].get_lines()) == 2 * lines
    assert len(disp2.axes_[0, 1].get_lines()) == 2 * lines