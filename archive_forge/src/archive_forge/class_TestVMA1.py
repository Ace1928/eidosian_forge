import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.statespace import varmax, sarimax
from statsmodels.iolib.summary import forg
from .results import results_varmax
class TestVMA1(CheckFREDManufacturing):
    """
    Test against the sspace VARMA example with some params set to zeros.
    """

    @classmethod
    def setup_class(cls):
        true = results_varmax.fred_vma1.copy()
        true['predict'] = varmax_results.iloc[1:][['predict_vma1_1', 'predict_vma1_2']]
        true['dynamic_predict'] = varmax_results.iloc[1:][['dyn_predict_vma1_1', 'dyn_predict_vma1_2']]
        super().setup_class(true, order=(0, 1), trend='n', error_cov_type='diagonal')

    def test_mle(self):
        pass

    @pytest.mark.skip('Known failure: standard errors do not match.')
    def test_bse_approx(self):
        pass

    @pytest.mark.skip('Known failure: standard errors do not match.')
    def test_bse_oim(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_predict(self):
        super().test_predict(end='2009-05-01', atol=0.0001)

    def test_dynamic_predict(self):
        super().test_dynamic_predict(end='2009-05-01', dynamic='2000-01-01')