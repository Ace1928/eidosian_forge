import pytest
import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
def test_user_specified_kernel(self):
    model = nparam.KernelReg(endog=[self.y], exog=[self.c1, self.c2], reg_type='ll', var_type='cc', bw='cv_ls', ckertype='tricube')
    sm_bw = model.bw
    R_bw = [0.581663, 0.5652]
    sm_mean, sm_mfx = model.fit()
    sm_mean = sm_mean[0:5]
    sm_mfx = sm_mfx[0:5]
    R_mean = [30.926714, 36.994604, 44.438358, 40.680598, 35.961593]
    sm_R2 = model.r_squared()
    R_R2 = 0.934825
    npt.assert_allclose(sm_bw, R_bw, atol=0.01)
    npt.assert_allclose(sm_mean, R_mean, atol=0.01)
    npt.assert_allclose(sm_R2, R_R2, atol=0.01)