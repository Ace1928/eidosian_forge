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
def test_process_multiplication(self):
    process1 = ArmaProcess.from_coeffs([0.9])
    process2 = ArmaProcess.from_coeffs([0.7])
    process3 = process1 * process2
    assert_equal(process3.arcoefs, np.array([1.6, -0.7 * 0.9]))
    assert_equal(process3.macoefs, np.array([]))
    process1 = ArmaProcess.from_coeffs([0.9], [0.2])
    process2 = ArmaProcess.from_coeffs([0.7])
    process3 = process1 * process2
    assert_equal(process3.arcoefs, np.array([1.6, -0.7 * 0.9]))
    assert_equal(process3.macoefs, np.array([0.2]))
    process1 = ArmaProcess.from_coeffs([0.9], [0.2])
    process2 = process1 * (np.array([1.0, -0.7]), np.array([1.0]))
    assert_equal(process2.arcoefs, np.array([1.6, -0.7 * 0.9]))
    assert_raises(TypeError, process1.__mul__, [3])