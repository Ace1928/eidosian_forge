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
class TestVAR_exog(CheckLutkepohl):

    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_exog.copy()
        true['predict'] = var_results.iloc[1:76][['predict_exog1_1', 'predict_exog1_2', 'predict_exog1_3']]
        true['predict'].iloc[0, :] = 0
        true['fcast'] = var_results.iloc[76:][['fcast_exog1_dln_inv', 'fcast_exog1_dln_inc', 'fcast_exog1_dln_consump']]
        exog = np.arange(75) + 2
        super().setup_class(true, order=(1, 0), trend='n', error_cov_type='unstructured', exog=exog, initialization='approximate_diffuse', loglikelihood_burn=1)

    def test_mle(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal() ** 0.5
        assert_allclose(bse[:-6] ** 2, self.true['var_oim'], atol=1e-05)

    def test_bse_oim(self):
        bse = self.results._cov_params_oim().diagonal() ** 0.5
        assert_allclose(bse[:-6] ** 2, self.true['var_oim'], atol=1e-05)

    def test_predict(self):
        super(CheckLutkepohl, self).test_predict(end='1978-10-01', atol=0.001)

    def test_dynamic_predict(self):
        pass

    def test_forecast(self):
        exog = (np.arange(75, 75 + 16) + 2)[:, np.newaxis]
        desired = self.results.forecast(steps=16, exog=exog)
        assert_allclose(desired, self.true['fcast'], atol=1e-06)

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']
        assert re.search('Model:.*VARX\\(1\\)', tables[0])
        for i in range(self.model.k_endog):
            offset = i * self.model.k_endog
            table = tables[i + 2]
            name = self.model.endog_names[i]
            assert re.search('Results for equation %s' % name, table)
            assert len(table.split('\n')) == 9
            assert re.search('L1.dln_inv +%.4f' % params[offset + 0], table)
            assert re.search('L1.dln_inc +%.4f' % params[offset + 1], table)
            assert re.search('L1.dln_consump +%.4f' % params[offset + 2], table)
            assert re.search('beta.x1 +' + forg(params[self.model._params_regression][i], prec=4), table)
        table = tables[-1]
        assert re.search('Error covariance matrix', table)
        assert len(table.split('\n')) == 11
        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert re.search('{} +{:.4f}'.format(names[i], params[i]), table)