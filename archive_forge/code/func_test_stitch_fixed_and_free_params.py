import numpy as np
import pytest
from numpy.testing import assert_allclose
from statsmodels.tsa.innovations.arma_innovations import arma_innovations
from statsmodels.tsa.arima.datasets.brockwell_davis_2002 import lake
from statsmodels.tsa.arima.estimators.hannan_rissanen import (
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from statsmodels.tools.tools import Bunch
@pytest.mark.parametrize('fixed_lags, free_lags, fixed_params, free_params, spec_lags, expected_all_params', [([], [], [], [], [], []), ([2], [], [0.2], [], [2], [0.2]), ([], [1], [], [0.2], [1], [0.2]), ([1], [3], [0.2], [-0.2], [1, 3], [0.2, -0.2]), ([3], [1, 2], [0.2], [0.3, -0.2], [1, 2, 3], [0.3, -0.2, 0.2]), ([3, 1], [2, 4], [0.3, 0.1], [0.5, 0.0], [1, 2, 3, 4], [0.1, 0.5, 0.3, 0.0]), ([3, 10], [1, 2], [0.2, 0.5], [0.3, -0.2], [1, 2, 3, 10], [0.3, -0.2, 0.2, 0.5]), ([3, 10], [1, 2], [0.2, 0.5], [0.3, -0.2], [3, 1, 10, 2], [0.2, 0.3, 0.5, -0.2])])
def test_stitch_fixed_and_free_params(fixed_lags, free_lags, fixed_params, free_params, spec_lags, expected_all_params):
    actual_all_params = _stitch_fixed_and_free_params(fixed_lags, fixed_params, free_lags, free_params, spec_lags)
    assert actual_all_params == expected_all_params