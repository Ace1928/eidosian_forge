import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative
def _kolmogn_Pomeranz(n, x, cdf=True):
    """Computes Pr(D_n <= d) using the Pomeranz recursion algorithm.

    Pomeranz (1974) [2]
    """
    t = n * x
    ll = int(np.floor(t))
    f = 1.0 * (t - ll)
    g = min(f, 1.0 - f)
    ceilf = 1 if f > 0 else 0
    roundf = 1 if f > 0.5 else 0
    npwrs = 2 * (ll + 1)
    gpower = np.empty(npwrs)
    twogpower = np.empty(npwrs)
    onem2gpower = np.empty(npwrs)
    gpower[0] = 1.0
    twogpower[0] = 1.0
    onem2gpower[0] = 1.0
    expnt = 0
    g_over_n, two_g_over_n, one_minus_two_g_over_n = (g / n, 2 * g / n, (1 - 2 * g) / n)
    for m in range(1, npwrs):
        gpower[m] = gpower[m - 1] * g_over_n / m
        twogpower[m] = twogpower[m - 1] * two_g_over_n / m
        onem2gpower[m] = onem2gpower[m - 1] * one_minus_two_g_over_n / m
    V0 = np.zeros([npwrs])
    V1 = np.zeros([npwrs])
    V1[0] = 1
    V0s, V1s = (0, 0)
    j1, j2 = _pomeranz_compute_j1j2(0, n, ll, ceilf, roundf)
    for i in range(1, 2 * n + 2):
        k1 = j1
        V0, V1 = (V1, V0)
        V0s, V1s = (V1s, V0s)
        V1.fill(0.0)
        j1, j2 = _pomeranz_compute_j1j2(i, n, ll, ceilf, roundf)
        if i == 1 or i == 2 * n + 1:
            pwrs = gpower
        else:
            pwrs = twogpower if i % 2 else onem2gpower
        ln2 = j2 - k1 + 1
        if ln2 > 0:
            conv = np.convolve(V0[k1 - V0s:k1 - V0s + ln2], pwrs[:ln2])
            conv_start = j1 - k1
            conv_len = j2 - j1 + 1
            V1[:conv_len] = conv[conv_start:conv_start + conv_len]
            if 0 < np.max(V1) < _EM128:
                V1 *= _EP128
                expnt -= _E128
            V1s = V0s + j1 - k1
    ans = V1[n - V1s]
    for m in range(1, n + 1):
        if np.abs(ans) > _EP128:
            ans *= _EM128
            expnt += _E128
        ans *= m
    if expnt != 0:
        ans = np.ldexp(ans, expnt)
    ans = _select_and_clip_prob(ans, 1.0 - ans, cdf)
    return ans