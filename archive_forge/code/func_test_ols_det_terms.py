import numpy as np
from numpy.testing import assert_, assert_allclose, assert_raises
import statsmodels.datasets.macrodata.data as macro
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.var_model import VAR
from .JMulTi_results.parse_jmulti_var_output import (
def test_ols_det_terms():
    if debug_mode:
        if 'det' not in to_test:
            return
        print('\n\nESTIMATED PARAMETERS FOR DETERMINISTIC TERMS', end='')
    for ds in datasets:
        for dt_s in dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt_s) + ': ', end='')
            err_msg = build_err_msg(ds, dt_s, 'PARAMETER MATRICES EXOG')
            det_key_ref = 'Deterministic term'
            if det_key_ref not in results_ref[ds][dt_s]['est'].keys():
                assert_(results_sm[ds][dt_s].coefs_exog.size == 0 and results_sm[ds][dt_s].stderr_dt.size == 0 and (results_sm[ds][dt_s].tvalues_dt.size == 0) and (results_sm[ds][dt_s].pvalues_dt.size == 0), err_msg)
                continue
            obtained = results_sm[ds][dt_s].coefs_exog
            desired = results_ref[ds][dt_s]['est'][det_key_ref]
            desired = reorder_jmultis_det_terms(desired, dt_s[0].startswith('c'), dt_s[1])
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if debug_mode and dont_test_se_t_p:
                continue
            obt = results_sm[ds][dt_s].stderr_dt
            des = results_ref[ds][dt_s]['se'][det_key_ref]
            des = reorder_jmultis_det_terms(des, dt_s[0].startswith('c'), dt_s[1]).T
            assert_allclose(obt, des, rtol, atol, False, 'STANDARD ERRORS\n' + err_msg)
            obt = results_sm[ds][dt_s].tvalues_dt
            des = results_ref[ds][dt_s]['t'][det_key_ref]
            des = reorder_jmultis_det_terms(des, dt_s[0].startswith('c'), dt_s[1]).T
            assert_allclose(obt, des, rtol, atol, False, 't-VALUES\n' + err_msg)
            obt = results_sm[ds][dt_s].pvalues_dt
            des = results_ref[ds][dt_s]['p'][det_key_ref]
            des = reorder_jmultis_det_terms(des, dt_s[0].startswith('c'), dt_s[1]).T
            assert_allclose(obt, des, rtol, atol, False, 'p-VALUES\n' + err_msg)