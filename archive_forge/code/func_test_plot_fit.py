import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
@pytest.mark.matplotlib
def test_plot_fit(self, close_figures):
    res = self.res
    fig = plot_fit(res, 0, y_true=None)
    x0 = res.model.exog[:, 0]
    yf = res.fittedvalues
    y = res.model.endog
    px1, px2 = fig.axes[0].get_lines()[0].get_data()
    np.testing.assert_equal(x0, px1)
    np.testing.assert_equal(y, px2)
    px1, px2 = fig.axes[0].get_lines()[1].get_data()
    np.testing.assert_equal(x0, px1)
    np.testing.assert_equal(yf, px2)
    close_or_save(pdf, fig)