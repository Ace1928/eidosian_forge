import pytest
import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
@pytest.mark.slow
def test_significance_discrete(self):
    nobs = 200
    np.random.seed(12345)
    ovals = np.random.binomial(2, 0.5, size=(nobs,))
    C2 = np.random.normal(2, 1, size=(nobs,))
    C3 = np.random.beta(0.5, 0.2, size=(nobs,))
    noise = np.random.normal(size=(nobs,))
    b1 = 1.2
    b2 = 3.7
    Y = b1 * ovals + b2 * C2 + noise
    bw = [3.63473198, 1214048.03]
    model = nparam.KernelReg(endog=[Y], exog=[ovals, C3], reg_type='ll', var_type='oc', bw=bw)
    nboot = 45
    sig_var1 = model.sig_test([0], nboot=nboot)
    npt.assert_equal(sig_var1 == 'Not Significant', False)
    sig_var2 = model.sig_test([1], nboot=nboot)
    npt.assert_equal(sig_var2 == 'Not Significant', True)