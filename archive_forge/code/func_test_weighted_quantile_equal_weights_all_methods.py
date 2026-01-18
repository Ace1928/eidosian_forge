from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.skip(reason='`method` argument is not currently exposed')
@pytest.mark.parametrize('da', ([1, 1.9, 2.2, 3, 3.7, 4.1, 5], [1, 1.9, 2.2, 3, 3.7, 4.1, np.nan], [np.nan, np.nan, np.nan]))
@pytest.mark.parametrize('q', (0.5, (0.2, 0.8)))
@pytest.mark.parametrize('skipna', (True, False))
@pytest.mark.parametrize('method', ['linear', 'interpolated_inverted_cdf', 'hazen', 'weibull', 'median_unbiased', 'normal_unbiased2'])
def test_weighted_quantile_equal_weights_all_methods(da, q, skipna, factor, method):
    da = DataArray(da)
    weights = xr.full_like(da, 3.14)
    expected = da.quantile(q, skipna=skipna, method=method)
    result = da.weighted(weights).quantile(q, skipna=skipna, method=method)
    assert_allclose(expected, result)