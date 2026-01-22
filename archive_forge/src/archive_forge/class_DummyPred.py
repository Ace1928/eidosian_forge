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
class DummyPred:

    def __init__(self, predicted_mean, row_labels):
        self.predicted_mean = predicted_mean
        self.row_labels = row_labels

        def f():
            raise NotImplementedError
        if not_implemented:
            self.forecast = property(f)