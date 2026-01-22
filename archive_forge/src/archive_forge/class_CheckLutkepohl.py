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
class CheckLutkepohl(CheckVARMAX):

    @classmethod
    def setup_class(cls, true, order, trend, error_cov_type, cov_type='approx', included_vars=['dln_inv', 'dln_inc', 'dln_consump'], **kwargs):
        cls.true = true
        dta = pd.DataFrame(results_varmax.lutkepohl_data, columns=['inv', 'inc', 'consump'], index=pd.date_range('1960-01-01', '1982-10-01', freq='QS'))
        dta['dln_inv'] = np.log(dta['inv']).diff()
        dta['dln_inc'] = np.log(dta['inc']).diff()
        dta['dln_consump'] = np.log(dta['consump']).diff()
        endog = dta.loc['1960-04-01':'1978-10-01', included_vars]
        cls.model = varmax.VARMAX(endog, order=order, trend=trend, error_cov_type=error_cov_type, **kwargs)
        cls.results = cls.model.smooth(true['params'], cov_type=cov_type)

    def test_predict(self, **kwargs):
        super().test_predict(end='1982-10-01', **kwargs)

    def test_dynamic_predict(self, **kwargs):
        super().test_dynamic_predict(end='1982-10-01', dynamic='1961-01-01', **kwargs)