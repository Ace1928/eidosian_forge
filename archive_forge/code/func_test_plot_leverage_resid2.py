import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
@pytest.mark.matplotlib
def test_plot_leverage_resid2(self, close_figures):
    fig = plot_leverage_resid2(self.res)
    assert_equal(isinstance(fig, plt.Figure), True)