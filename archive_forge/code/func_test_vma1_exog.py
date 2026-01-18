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
def test_vma1_exog():
    dta = pd.DataFrame(results_varmax.lutkepohl_data, columns=['inv', 'inc', 'consump'], index=pd.date_range('1960-01-01', '1982-10-01', freq='QS'))
    dta = np.log(dta).diff().iloc[1:]
    endog = dta.iloc[:, :2]
    exog = dta.iloc[:, 2]
    ma_params1 = [-0.01, 1.4, -0.3, 0.002]
    ma_params2 = [0.004, 0.8, -0.5, 0.0001]
    vma_params = [ma_params1[0], ma_params2[0], ma_params1[2], 0, 0, ma_params2[2], ma_params1[1], ma_params2[1], ma_params1[3], ma_params2[3]]
    mod_vma = varmax.VARMAX(endog, exog=exog, order=(0, 1), error_cov_type='diagonal')
    mod_vma.ssm.initialize_diffuse()
    res_mva = mod_vma.smooth(vma_params)
    sp = mod_vma.start_params
    assert_equal(len(sp), len(mod_vma.param_names))
    mod_ma1 = sarimax.SARIMAX(endog.iloc[:, 0], exog=exog, order=(0, 0, 1), trend='c')
    mod_ma1.ssm.initialize_diffuse()
    mod_ma2 = sarimax.SARIMAX(endog.iloc[:, 1], exog=exog, order=(0, 0, 1), trend='c')
    mod_ma2.ssm.initialize_diffuse()
    res_ma1 = mod_ma1.smooth(ma_params1)
    res_ma2 = mod_ma2.smooth(ma_params2)
    assert_allclose(res_mva.llf_obs[2:], (res_ma1.llf_obs + res_ma2.llf_obs)[2:])