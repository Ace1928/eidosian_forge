import pytest
import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
@pytest.mark.slow
def test_significance_continuous(self):
    nobs = 250
    np.random.seed(12345)
    C1 = np.random.normal(size=(nobs,))
    C2 = np.random.normal(2, 1, size=(nobs,))
    C3 = np.random.beta(0.5, 0.2, size=(nobs,))
    noise = np.random.normal(size=(nobs,))
    b1 = 1.2
    b2 = 3.7
    Y = b1 * C1 + b2 * C2 + noise
    bw = [11108137.1087194, 1333821.85150218]
    model = nparam.KernelReg(endog=[Y], exog=[C1, C3], reg_type='ll', var_type='cc', bw=bw)
    nboot = 45
    sig_var12 = model.sig_test([0, 1], nboot=nboot)
    npt.assert_equal(sig_var12 == 'Not Significant', False)
    sig_var1 = model.sig_test([0], nboot=nboot)
    npt.assert_equal(sig_var1 == 'Not Significant', False)
    sig_var2 = model.sig_test([1], nboot=nboot)
    npt.assert_equal(sig_var2 == 'Not Significant', True)