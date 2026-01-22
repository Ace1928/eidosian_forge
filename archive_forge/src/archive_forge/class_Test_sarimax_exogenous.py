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
class Test_sarimax_exogenous(SARIMAXCoverageTest):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3, 2, 2)
        kwargs['seasonal_order'] = (3, 2, 2, 4)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog)) ** 2
        super().setup_class(50, *args, **kwargs)

    def test_results_params(self):
        result = self.model.filter(self.true_params)
        assert_allclose(self.true_params[1:4], result.arparams)
        assert_allclose(self.true_params[4:6], result.maparams)
        assert_allclose(self.true_params[6:9], result.seasonalarparams)
        assert_allclose(self.true_params[9:11], result.seasonalmaparams)