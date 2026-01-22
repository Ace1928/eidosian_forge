import numpy as np
from scipy import stats
from statsmodels.tools.numdiff import _approx_fprime_cs_scalar, approx_hess
def _derivs(self, t, lamda, order=(1, 2)):
    if not isinstance(order, (int, np.integer)):
        if 1 in order and 2 in order:
            order = -1
        else:
            raise ValueError('order should be 1, 2, or (1,2)')
    dn = 1 / np.sqrt(2 * np.pi)
    a = lamda
    g = np.log((1.0 - t) / t) * 0.5 / a
    gd1 = 1 / (2 * a * (t - 1) * t)
    gd2 = (0.5 - t) / (a * ((1 - t) * t) ** 2)
    tp = a + g
    fp = stats.norm.cdf(tp)
    fd1p = np.exp(-tp ** 2 / 2) * dn
    fd2p = -fd1p * tp
    tn = a - g
    fn = stats.norm.cdf(tn)
    fd1n = np.exp(-tn ** 2 / 2) * dn
    fd2n = -fd1n * tn
    if order in (1, -1):
        d1 = gd1 * (-t * fd1n - (t - 1) * fd1p) + fn - fp
    if order in (2, -1):
        d2 = gd1 ** 2 * (t * fd2n - (t - 1) * fd2p) + (-(t - 1) * gd2 - 2 * gd1) * fd1p - (t * gd2 + 2 * gd1) * fd1n
    if order == 1:
        return d1
    elif order == 2:
        return d2
    elif order == -1:
        return (d1, d2)