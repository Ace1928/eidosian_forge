import functools
import warnings
import numpy as np
from ase.utils import IOContext
def _bandgap(e_skn, spin, direct):
    """Helper function."""
    ns, nk, nb = e_skn.shape
    s1 = s2 = k1 = k2 = n1 = n2 = None
    N_sk = (e_skn < 0.0).sum(2)
    if ns == 1:
        if N_sk[0].ptp() > 0:
            return (0.0, (None, None, None), (None, None, None))
    elif spin is None:
        if (N_sk.ptp(axis=1) > 0).any():
            return (0.0, (None, None, None), (None, None, None))
    elif N_sk[spin].ptp() > 0:
        return (0.0, (None, None, None), (None, None, None))
    if (N_sk == 0).any() or (N_sk == nb).any():
        raise ValueError('Too few bands!')
    e_skn = np.array([[e_skn[s, k, N_sk[s, k] - 1:N_sk[s, k] + 1] for k in range(nk)] for s in range(ns)])
    ev_sk = e_skn[:, :, 0]
    ec_sk = e_skn[:, :, 1]
    if ns == 1:
        s1 = 0
        s2 = 0
        gap, k1, k2 = find_gap(ev_sk[0], ec_sk[0], direct)
        n1 = N_sk[0, 0] - 1
        n2 = n1 + 1
        return (gap, (0, k1, n1), (0, k2, n2))
    if spin is None:
        gap, k1, k2 = find_gap(ev_sk.ravel(), ec_sk.ravel(), direct)
        if direct:
            for s in [0, 1]:
                g, k, _ = find_gap(ev_sk[s], ec_sk[1 - s], direct)
                if g < gap:
                    gap = g
                    k1 = k + nk * s
                    k2 = k + nk * (1 - s)
        if gap > 0.0:
            s1, k1 = divmod(k1, nk)
            s2, k2 = divmod(k2, nk)
            n1 = N_sk[s1, k1] - 1
            n2 = N_sk[s2, k2]
            return (gap, (s1, k1, n1), (s2, k2, n2))
        return (0.0, (None, None, None), (None, None, None))
    gap, k1, k2 = find_gap(ev_sk[spin], ec_sk[spin], direct)
    s1 = spin
    s2 = spin
    n1 = N_sk[s1, k1] - 1
    n2 = n1 + 1
    return (gap, (s1, k1, n1), (s2, k2, n2))