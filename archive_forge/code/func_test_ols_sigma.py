import numpy as np
from numpy.testing import assert_, assert_allclose, assert_raises
import statsmodels.datasets.macrodata.data as macro
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.var_model import VAR
from .JMulTi_results.parse_jmulti_var_output import (
def test_ols_sigma():
    if debug_mode:
        if 'Sigma_u' not in to_test:
            return
        print('\n\nSIGMA_U', end='')
    for ds in datasets:
        for dt in dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            err_msg = build_err_msg(ds, dt, 'Sigma_u')
            obtained = results_sm[ds][dt].sigma_u
            desired = results_ref[ds][dt]['est']['Sigma_u']
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)