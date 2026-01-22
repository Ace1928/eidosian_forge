import numpy as np
import numpy.testing as npt
from statsmodels.sandbox.nonparametric.kernel_extras import SemiLinear
class TestSemiLinear(KernelExtrasTestBase):

    def test_basic(self):
        nobs = 300
        np.random.seed(1234)
        C1 = np.random.normal(0, 2, size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        e = np.random.normal(size=(nobs,))
        b1 = 1.3
        b2 = -0.7
        Y = b1 * C1 + np.exp(b2 * C2) + e
        model = SemiLinear(endog=[Y], exog=[C1], exog_nonparametric=[C2], var_type='c', k_linear=1)
        b_hat = np.squeeze(model.b)
        npt.assert_allclose(b1, b_hat, rtol=0.1)