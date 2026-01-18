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
def test_sarimax_starting_values_few_obsevations(reset_randomstate):
    y = np.random.standard_normal(17)
    sarimax_model = sarimax.SARIMAX(endog=y, order=(1, 1, 1), seasonal_order=(0, 1, 0, 12), trend='n').fit(disp=False)
    assert np.all(np.isfinite(sarimax_model.predict(start=len(y), end=len(y) + 11)))