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
class TestVAR_measurement_error(CheckLutkepohl):
    """
    Notes
    -----
    There does not appear to be a way to get Stata to estimate a VAR with
    measurement errors. Thus this test is mostly a smoke test that measurement
    errors are setup correctly: it uses the same params from TestVAR_diagonal
    and sets the measurement errors variance params to zero to check that the
    loglike and predict are the same.

    It also checks that the state-space representation with positive
    measurement errors is correct.
    """

    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_diag_meas.copy()
        true['predict'] = var_results.iloc[1:][['predict_diag1', 'predict_diag2', 'predict_diag3']]
        true['dynamic_predict'] = var_results.iloc[1:][['dyn_predict_diag1', 'dyn_predict_diag2', 'dyn_predict_diag3']]
        super().setup_class(true, order=(1, 0), trend='n', error_cov_type='diagonal', measurement_error=True)
        cls.true_measurement_error_variances = [1.0, 2.0, 3.0]
        params = np.r_[true['params'][:-3], cls.true_measurement_error_variances]
        cls.results2 = cls.model.smooth(params)

    def test_mle(self):
        pass

    def test_bse_approx(self):
        pass

    def test_bse_oim(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_representation(self):
        for name in self.model.ssm.shapes.keys():
            if name == 'obs':
                pass
            elif name == 'obs_cov':
                actual = self.results2.filter_results.obs_cov
                desired = np.diag(self.true_measurement_error_variances)[:, :, np.newaxis]
                assert_equal(actual, desired)
            else:
                assert_equal(getattr(self.results2.filter_results, name), getattr(self.results.filter_results, name))

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']
        assert re.search('Model:.*VAR\\(1\\)', tables[0])
        for i in range(self.model.k_endog):
            offset = i * self.model.k_endog
            table = tables[i + 2]
            name = self.model.endog_names[i]
            assert re.search('Results for equation %s' % name, table)
            assert len(table.split('\n')) == 9
            assert re.search('L1.dln_inv +%.4f' % params[offset + 0], table)
            assert re.search('L1.dln_inc +%.4f' % params[offset + 1], table)
            assert re.search('L1.dln_consump +%.4f' % params[offset + 2], table)
            assert re.search('measurement_variance +%.4g' % params[-(i + 1)], table)
        table = tables[-1]
        assert re.search('Error covariance matrix', table)
        assert len(table.split('\n')) == 8
        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert re.search('{} +{:.4f}'.format(names[i], params[i]), table)