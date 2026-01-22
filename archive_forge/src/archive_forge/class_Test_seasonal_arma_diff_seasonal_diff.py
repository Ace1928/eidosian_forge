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
class Test_seasonal_arma_diff_seasonal_diff(SARIMAXCoverageTest):

    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0, 2, 0)
        kwargs['seasonal_order'] = (3, 2, 2, 4)
        super().setup_class(47, *args, **kwargs)

    def test_results(self):
        self.result = self.model.filter(self.true_params)
        self.result.summary()
        self.result.cov_params_default
        self.result.cov_params_oim
        self.result.cov_params_opg