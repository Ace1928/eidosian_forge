import pytest
import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
def test_ordered_lc_cvls(self):
    model = nparam.KernelReg(endog=[self.Italy_gdp], exog=[self.Italy_year], reg_type='lc', var_type='o', bw='cv_ls')
    sm_bw = model.bw
    R_bw = 0.1390096
    sm_mean, sm_mfx = model.fit()
    sm_mean = sm_mean[0:5]
    sm_mfx = sm_mfx[0:5]
    R_mean = 6.190486
    sm_R2 = model.r_squared()
    R_R2 = 0.1435323
    npt.assert_allclose(sm_bw, R_bw, atol=0.01)
    npt.assert_allclose(sm_mean, R_mean, atol=0.01)
    npt.assert_allclose(sm_R2, R_R2, atol=0.01)