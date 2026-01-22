import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from statsmodels.iolib.summary import forg
class TestDynamicFactor_exog2(CheckDynamicFactor):
    """
    Test for a dynamic factor model with 2 exogenous regressors: a constant
    and a time-trend
    """

    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_dfm_exog2.copy()
        true['predict'] = output_results.iloc[1:][['predict_dfm_exog2_1', 'predict_dfm_exog2_2', 'predict_dfm_exog2_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_dfm_exog2_1', 'dyn_predict_dfm_exog2_2', 'dyn_predict_dfm_exog2_3']]
        exog = np.c_[np.ones((75, 1)), (np.arange(75) + 2)[:, np.newaxis]]
        super().setup_class(true, k_factors=1, factor_order=1, exog=exog)

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal() ** 0.5
        assert_allclose(bse ** 2, self.true['var_oim'], atol=1e-05)

    def test_predict(self):
        exog = np.c_[np.ones((16, 1)), (np.arange(75, 75 + 16) + 2)[:, np.newaxis]]
        super().test_predict(exog=exog)

    def test_dynamic_predict(self):
        exog = np.c_[np.ones((16, 1)), (np.arange(75, 75 + 16) + 2)[:, np.newaxis]]
        super().test_dynamic_predict(exog=exog)

    def test_summary(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']
        assert_equal(len(tables), 2 + self.model.k_endog + self.model.k_factors + 1)
        assert re.search('Model:.*DynamicFactor\\(factors=1, order=1\\)', tables[0])
        assert_equal(re.search('.*2 regressors', tables[0]) is None, False)
        for i in range(self.model.k_endog):
            offset_loading = self.model.k_factors * i
            offset_exog = self.model.k_factors * self.model.k_endog
            table = tables[i + 2]
            name = self.model.endog_names[i]
            assert re.search('Results for equation %s' % name, table)
            assert_equal(len(table.split('\n')), 8)
            assert re.search('loading.f1 +' + forg(params[offset_loading + 0], prec=4), table)
            assert re.search('beta.const +' + forg(params[offset_exog + i * 2 + 0], prec=4), table)
            assert re.search('beta.x1 +' + forg(params[offset_exog + i * 2 + 1], prec=4), table)
        for i in range(self.model.k_factors):
            offset = self.model.k_endog * (self.model.k_factors + 3) + i * self.model.k_factors
            table = tables[self.model.k_endog + i + 2]
            name = self.model.endog_names[i]
            assert re.search('Results for factor equation f%d' % (i + 1), table)
            assert_equal(len(table.split('\n')), 6)
            assert re.search('L1.f1 +' + forg(params[offset + 0], prec=4), table)
        table = tables[2 + self.model.k_endog + self.model.k_factors]
        name = self.model.endog_names[i]
        assert re.search('Error covariance matrix', table)
        assert_equal(len(table.split('\n')), 8)
        offset = self.model.k_endog * (self.model.k_factors + 2)
        for i in range(self.model.k_endog):
            iname = self.model.endog_names[i]
            iparam = forg(params[offset + i], prec=4)
            assert re.search('sigma2.{} +{}'.format(iname, iparam), table)