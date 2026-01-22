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
class TestVARMA(CheckFREDManufacturing):
    """
    Test against the sspace VARMA example with some params set to zeros.
    """

    @classmethod
    def setup_class(cls):
        true = results_varmax.fred_varma11.copy()
        true['predict'] = varmax_results.iloc[1:][['predict_varma11_1', 'predict_varma11_2']]
        true['dynamic_predict'] = varmax_results.iloc[1:][['dyn_predict_varma11_1', 'dyn_predict_varma11_2']]
        super().setup_class(true, order=(1, 1), trend='n', error_cov_type='diagonal')

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

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']
        assert re.search('Model:.*VARMA\\(1,1\\)', tables[0])
        for i in range(self.model.k_endog):
            offset_ar = i * self.model.k_endog
            offset_ma = self.model.k_endog ** 2 * self.model.k_ar + i * self.model.k_endog
            table = tables[i + 2]
            name = self.model.endog_names[i]
            assert re.search('Results for equation %s' % name, table)
            assert len(table.split('\n')) == 9
            assert re.search('L1.dlncaputil +' + forg(params[offset_ar + 0], prec=4), table)
            assert re.search('L1.dlnhours +' + forg(params[offset_ar + 1], prec=4), table)
            assert re.search('L1.e\\(dlncaputil\\) +' + forg(params[offset_ma + 0], prec=4), table)
            assert re.search('L1.e\\(dlnhours\\) +' + forg(params[offset_ma + 1], prec=4), table)
        table = tables[-1]
        assert re.search('Error covariance matrix', table)
        assert len(table.split('\n')) == 7
        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert re.search('{} +{}'.format(names[i], forg(params[i], prec=4)), table)