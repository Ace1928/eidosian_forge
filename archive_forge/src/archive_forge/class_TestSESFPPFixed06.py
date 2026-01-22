import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class TestSESFPPFixed06(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(oildata, initialization_method='simple')
        res = mod.filter([results_params['oil_fpp2']['alpha']])
        super().setup_class('oil_fpp2', res)