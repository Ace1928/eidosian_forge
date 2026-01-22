import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestHoltDampedFPPEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True, damped_trend=True, concentrate_scale=False)
        params = [results_params['air_fpp2']['alpha'], results_params['air_fpp2']['beta'], results_params['air_fpp2']['phi'], results_params['air_fpp2']['sigma2'], results_params['air_fpp2']['l0'], results_params['air_fpp2']['b0']]
        res = mod.filter(params)
        super().setup_class('air_fpp2', res)