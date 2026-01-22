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
class TestRealGDPARStata:
    """
    Includes tests of filtered states and standardized forecast errors.

    Notes
    -----
    Could also test the usual things like standard errors, etc. but those are
    well-tested elsewhere.
    """

    @classmethod
    def setup_class(cls):
        dlgdp = np.log(realgdp_results['value']).diff()[1:].values
        cls.model = sarimax.SARIMAX(dlgdp, order=(12, 0, 0), trend='n', hamilton_representation=True)
        params = [0.40725515, 0.18782621, -0.01514009, -0.01027267, -0.03642297, 0.11576416, 0.02573029, -0.00766572, 0.13506498, 0.08649569, 0.06942822, -0.10685783, 7.999607e-05]
        cls.results = cls.model.filter(params)

    def test_filtered_state(self):
        for i in range(12):
            assert_allclose(realgdp_results.iloc[1:]['u%d' % (i + 1)], self.results.filter_results.filtered_state[i], atol=1e-06)

    def test_standardized_forecasts_error(self):
        assert_allclose(realgdp_results.iloc[1:]['rstd'], self.results.filter_results.standardized_forecasts_error[0], atol=0.001)