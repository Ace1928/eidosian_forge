import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (

    ARMAX model: Friedman quantity theory of money, forecasts

    Variation on:
    Stata arima postestimation documentation, Example 1 - Dynamic forecasts

    This is a variation of the Stata example, in which the endogenous data is
    actually made to be missing so that the predict command must forecast.

    As another unit test, we also compare against the case in State when
    predict is used against missing data (so forecasting) with the dynamic
    option also included. Note, however, that forecasting in State space models
    amounts to running the Kalman filter against missing datapoints, so it is
    not clear whether "dynamic" forecasting (where instead of missing
    datapoints for lags, we plug in previous forecasted endog values) is
    meaningful.
    