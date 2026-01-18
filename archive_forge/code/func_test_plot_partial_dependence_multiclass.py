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
def test_plot_partial_dependence_multiclass(pyplot):
    grid_resolution = 25
    clf_int = GradientBoostingClassifier(n_estimators=10, random_state=1)
    iris = load_iris()
    clf_int.fit(iris.data, iris.target)
    disp_target_0 = PartialDependenceDisplay.from_estimator(clf_int, iris.data, [0, 3], target=0, grid_resolution=grid_resolution)
    assert disp_target_0.figure_ is pyplot.gcf()
    assert disp_target_0.axes_.shape == (1, 2)
    assert disp_target_0.lines_.shape == (1, 2)
    assert disp_target_0.contours_.shape == (1, 2)
    assert disp_target_0.deciles_vlines_.shape == (1, 2)
    assert disp_target_0.deciles_hlines_.shape == (1, 2)
    assert all((c is None for c in disp_target_0.contours_.flat))
    assert disp_target_0.target_idx == 0
    target = iris.target_names[iris.target]
    clf_symbol = GradientBoostingClassifier(n_estimators=10, random_state=1)
    clf_symbol.fit(iris.data, target)
    disp_symbol = PartialDependenceDisplay.from_estimator(clf_symbol, iris.data, [0, 3], target='setosa', grid_resolution=grid_resolution)
    assert disp_symbol.figure_ is pyplot.gcf()
    assert disp_symbol.axes_.shape == (1, 2)
    assert disp_symbol.lines_.shape == (1, 2)
    assert disp_symbol.contours_.shape == (1, 2)
    assert disp_symbol.deciles_vlines_.shape == (1, 2)
    assert disp_symbol.deciles_hlines_.shape == (1, 2)
    assert all((c is None for c in disp_symbol.contours_.flat))
    assert disp_symbol.target_idx == 0
    for int_result, symbol_result in zip(disp_target_0.pd_results, disp_symbol.pd_results):
        assert_allclose(int_result.average, symbol_result.average)
        assert_allclose(int_result['grid_values'], symbol_result['grid_values'])
    disp_target_1 = PartialDependenceDisplay.from_estimator(clf_int, iris.data, [0, 3], target=1, grid_resolution=grid_resolution)
    target_0_data_y = disp_target_0.lines_[0, 0].get_data()[1]
    target_1_data_y = disp_target_1.lines_[0, 0].get_data()[1]
    assert any(target_0_data_y != target_1_data_y)