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
class Friedman(SARIMAXStataTests):
    """
    ARMAX model: Friedman quantity theory of money

    Stata arima documentation, Example 4
    """

    @classmethod
    def setup_class(cls, true, exog=None, *args, **kwargs):
        cls.true = true
        endog = np.r_[true['data']['consump']]
        if exog is None:
            exog = add_constant(true['data']['m2'])
        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)
        cls.model = sarimax.SARIMAX(endog, *args, exog=exog, order=(1, 0, 1), **kwargs)
        params = np.r_[true['params_exog'], true['params_ar'], true['params_ma'], true['params_variance']]
        cls.result = cls.model.filter(params)