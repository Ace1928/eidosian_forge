from statsmodels.compat.pandas import MONTH_END, QUARTER_END, YEAR_END
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import seasonal_decompose
@pytest.mark.matplotlib
@pytest.mark.parametrize('model', ['additive', 'multiplicative'])
@pytest.mark.parametrize('freq', [4, 12])
@pytest.mark.parametrize('two_sided', [True, False])
@pytest.mark.parametrize('extrapolate_trend', [True, False])
def test_seasonal_decompose_plot(model, freq, two_sided, extrapolate_trend):
    x = np.array([-50, 175, 149, 214, 247, 237, 225, 329, 729, 809, 530, 489, 540, 457, 195, 176, 337, 239, 128, 102, 232, 429, 3, 98, 43, -141, -77, -13, 125, 361, -45, 184])
    x -= x.min() + 1
    x2 = np.r_[x[12:], x[:12]]
    x = np.c_[x, x2]
    res = seasonal_decompose(x, period=freq, two_sided=two_sided, extrapolate_trend=extrapolate_trend)
    res.plot()