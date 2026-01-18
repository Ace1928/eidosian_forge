import os
import warnings
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.simulation_smoother import SimulationSmoother
from statsmodels.tsa.statespace import tools, sarimax
from .results import results_kalman_filter
from numpy.testing import (
def test_standardized_forecasts_error():
    true = results_kalman_filter.uc_uni
    data = pd.DataFrame(true['data'], index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'), columns=['GDP'])
    data['lgdp'] = np.log(data['GDP'])
    mod = sarimax.SARIMAX(data['lgdp'], order=(1, 1, 0), use_exact_diffuse=True)
    res = mod.fit(disp=-1)
    d = np.maximum(res.loglikelihood_burn, res.nobs_diffuse)
    standardized_forecasts_error = res.filter_results.forecasts_error[0] / np.sqrt(res.filter_results.forecasts_error_cov[0, 0])
    assert_allclose(res.filter_results.standardized_forecasts_error[0, d:], standardized_forecasts_error[..., d:])