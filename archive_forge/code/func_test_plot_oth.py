import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
@pytest.mark.matplotlib
def test_plot_oth(self, close_figures):
    res = self.res
    plot_fit(res, 0, y_true=None)
    plot_partregress_grid(res, exog_idx=[0, 1])
    plot_partregress_grid(self.res_true, grid=(2, 3))
    plot_regress_exog(res, exog_idx=0)
    plot_ccpr(res, exog_idx=0)
    plot_ccpr_grid(res, exog_idx=[0])
    fig = plot_ccpr_grid(res, exog_idx=[0, 1])
    for ax in fig.axes:
        add_lowess(ax)
    close_or_save(pdf, fig)