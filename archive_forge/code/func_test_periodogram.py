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
def test_periodogram(self):
    process = ArmaProcess()
    pg = process.periodogram()
    assert_almost_equal(pg[0], np.linspace(0, np.pi, 100, False))
    assert_almost_equal(pg[1], np.sqrt(2 / np.pi) / 2 * np.ones(100))