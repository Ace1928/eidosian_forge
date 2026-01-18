from statsmodels.compat.pandas import MONTH_END, QUARTER_END, YEAR_END
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import seasonal_decompose
def test_one_sided_moving_average_in_stl_decompose(self):
    res_add = seasonal_decompose(self.data.values, period=4, two_sided=False)
    seasonal = np.array([76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4, 76.76, 90.03, -114.4, -52.4])
    trend = np.array([np.nan, np.nan, np.nan, np.nan, 159.12, 204.0, 221.25, 245.12, 319.75, 451.5, 561.12, 619.25, 615.62, 548.0, 462.12, 381.12, 316.62, 264.0, 228.38, 210.75, 188.38, 199.0, 207.12, 191.0, 166.88, 72.0, -9.25, -33.12, -36.75, 36.25, 103.0, 131.62])
    resid = np.array([np.nan, np.nan, np.nan, np.nan, 11.112, -57.031, 118.147, 136.272, 332.487, 267.469, 83.272, -77.853, -152.388, -181.031, -152.728, -152.728, -56.388, -115.031, 14.022, -56.353, -33.138, 139.969, -89.728, -40.603, -200.638, -303.031, 46.647, 72.522, 84.987, 234.719, -33.603, 104.772])
    assert_almost_equal(res_add.seasonal, seasonal, 2)
    assert_almost_equal(res_add.trend, trend, 2)
    assert_almost_equal(res_add.resid, resid, 3)
    res_mult = seasonal_decompose(np.abs(self.data.values), 'm', period=4, two_sided=False)
    seasonal = np.array([1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755, 1.1985, 1.5449, 0.5811, 0.6755])
    trend = np.array([np.nan, np.nan, np.nan, np.nan, 171.625, 204.0, 221.25, 245.125, 319.75, 451.5, 561.125, 619.25, 615.625, 548.0, 462.125, 381.125, 316.625, 264.0, 228.375, 210.75, 188.375, 199.0, 207.125, 191.0, 166.875, 107.25, 80.5, 79.125, 78.75, 116.5, 140.0, 157.375])
    resid = np.array([np.nan, np.nan, np.nan, np.nan, 1.2008, 0.752, 1.75, 1.987, 1.9023, 1.1598, 1.6253, 1.169, 0.7319, 0.5398, 0.7261, 0.6837, 0.888, 0.586, 0.9645, 0.7165, 1.0276, 1.3954, 0.0249, 0.7596, 0.215, 0.851, 1.646, 0.2432, 1.3244, 2.0058, 0.5531, 1.7309])
    assert_almost_equal(res_mult.seasonal, seasonal, 4)
    assert_almost_equal(res_mult.trend, trend, 2)
    assert_almost_equal(res_mult.resid, resid, 4)
    res_add = seasonal_decompose(self.data.values[:-1], period=4, two_sided=False)
    seasonal = np.array([81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95, -65.74, 81.21, 94.48, -109.95])
    trend = [np.nan, np.nan, np.nan, np.nan, 159.12, 204.0, 221.25, 245.12, 319.75, 451.5, 561.12, 619.25, 615.62, 548.0, 462.12, 381.12, 316.62, 264.0, 228.38, 210.75, 188.38, 199.0, 207.12, 191.0, 166.88, 72.0, -9.25, -33.12, -36.75, 36.25, 103.0]
    random = [np.nan, np.nan, np.nan, np.nan, 6.663, -61.48, 113.699, 149.618, 328.038, 263.02, 78.824, -64.507, -156.837, -185.48, -157.176, -139.382, -60.837, -119.48, 9.574, -43.007, -37.587, 135.52, -94.176, -27.257, -205.087, -307.48, 42.199, 85.868, 80.538, 230.27, -38.051]
    assert_almost_equal(res_add.seasonal, seasonal, 2)
    assert_almost_equal(res_add.trend, trend, 2)
    assert_almost_equal(res_add.resid, random, 3)