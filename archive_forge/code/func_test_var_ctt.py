import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from statsmodels.tsa.statespace import varmax
from .results import results_var_R
def test_var_ctt():
    test = 'ctt_as_exog1'
    results = results_var_R.res_ctt_as_exog1
    mod = varmax.VARMAX(endog, order=(2, 0), trend='ctt', loglikelihood_burn=2)
    params = results['params']
    params = np.r_[params[-(6 + 9):-6], params[:-(6 + 9)], params[-6:]]
    res = mod.smooth(params)
    assert_allclose(res.llf, results['llf'])
    columns = ['{}.fcast.{}.fcst'.format(test, name) for name in endog.columns]
    assert_allclose(res.forecast(10), results_var_R_output[columns].iloc[:10])
    check_irf(test, mod, results, params)