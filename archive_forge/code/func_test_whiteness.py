import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
def test_whiteness():
    if debug_mode:
        if 'whiteness' not in to_test:
            return
        else:
            print('\n\nTEST WHITENESS OF RESIDUALS', end='')
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None
            lags = results_ref[ds][dt]['whiteness']['tested order']
            obtained = results_sm[ds][dt].test_whiteness(nlags=lags)
            obtained_exog = results_sm_exog[ds][dt].test_whiteness(nlags=lags)
            obtained_exog_coint = results_sm_exog_coint[ds][dt].test_whiteness(nlags=lags)
            err_msg = build_err_msg(ds, dt, 'WHITENESS OF RESIDUALS - TEST STATISTIC')
            desired = results_ref[ds][dt]['whiteness']['test statistic']
            assert_allclose(obtained.test_statistic, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog.test_statistic, obtained.test_statistic, 'WITH EXOG' + err_msg)
            if exog_coint:
                assert_equal(obtained_exog_coint.test_statistic, obtained.test_statistic, 'WITH EXOG_COINT' + err_msg)
            err_msg = build_err_msg(ds, dt, 'WHITENESS OF RESIDUALS - P-VALUE')
            desired = results_ref[ds][dt]['whiteness']['p-value']
            assert_allclose(obtained.pvalue, desired, rtol, atol, False, err_msg)
            obtained = results_sm[ds][dt].test_whiteness(nlags=lags, adjusted=True)
            obtained_exog = results_sm_exog[ds][dt].test_whiteness(nlags=lags, adjusted=True)
            obtained_exog_coint = results_sm_exog_coint[ds][dt].test_whiteness(nlags=lags, adjusted=True)
            err_msg = build_err_msg(ds, dt, 'WHITENESS OF RESIDUALS - TEST STATISTIC (ADJUSTED TEST)')
            desired = results_ref[ds][dt]['whiteness']['test statistic adj.']
            assert_allclose(obtained.test_statistic, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog.test_statistic, obtained.test_statistic, 'WITH EXOG' + err_msg)
            if exog_coint:
                assert_equal(obtained_exog_coint.test_statistic, obtained.test_statistic, 'WITH EXOG_COINT' + err_msg)
            err_msg = build_err_msg(ds, dt, 'WHITENESS OF RESIDUALS - P-VALUE (ADJUSTED TEST)')
            desired = results_ref[ds][dt]['whiteness']['p-value adjusted']
            assert_allclose(obtained.pvalue, desired, rtol, atol, False, err_msg)
            obtained.summary()
            str(obtained)
            assert_(obtained == obtained_exog)