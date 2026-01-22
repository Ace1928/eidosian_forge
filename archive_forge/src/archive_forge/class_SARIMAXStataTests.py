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
class SARIMAXStataTests:

    def test_loglike(self):
        assert_almost_equal(self.result.llf, self.true['loglike'], 4)

    def test_aic(self):
        assert_almost_equal(self.result.aic, self.true['aic'], 3)

    def test_bic(self):
        assert_almost_equal(self.result.bic, self.true['bic'], 3)

    def test_hqic(self):
        hqic = -2 * self.result.llf + 2 * np.log(np.log(self.result.nobs_effective)) * self.result.params.shape[0]
        assert_almost_equal(self.result.hqic, hqic, 3)

    def test_standardized_forecasts_error(self):
        cython_sfe = self.result.standardized_forecasts_error
        self.result._standardized_forecasts_error = None
        python_sfe = self.result.standardized_forecasts_error
        assert_allclose(cython_sfe, python_sfe)