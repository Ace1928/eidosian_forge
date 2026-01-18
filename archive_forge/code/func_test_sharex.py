from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import statsmodels.datasets
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import Fourier
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tsa.statespace.exponential_smoothing import (
@pytest.mark.matplotlib
def test_sharex(data):
    stlf = STLForecast(data, ARIMA, model_kwargs={'order': (2, 0, 0)})
    res = stlf.fit(fit_kwargs={})
    plt = res.result.plot()
    grouper_view = plt.axes[0].get_shared_x_axes()
    sibs = grouper_view.get_siblings(plt.axes[1])
    assert len(sibs) == 4