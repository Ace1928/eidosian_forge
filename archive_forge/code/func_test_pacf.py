from statsmodels.compat.pandas import QUARTER_END
import datetime as dt
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.sandbox.tsa.fftarma import ArmaFft
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import (
from statsmodels.tsa.tests.results import results_arma_acf
from statsmodels.tsa.tests.results.results_process import (
def test_pacf(self):
    process1 = ArmaProcess.from_coeffs([0.9])
    pacf = process1.pacf(10)
    expected = np.array([1, 0.9] + [0] * 8)
    assert_array_almost_equal(pacf, expected)
    pacf = process1.pacf()
    assert_(pacf.shape[0] == process1.nobs)