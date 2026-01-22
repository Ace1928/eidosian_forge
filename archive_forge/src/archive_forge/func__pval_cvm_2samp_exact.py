from collections import namedtuple
from dataclasses import dataclass
from math import comb
import numpy as np
import warnings
from itertools import combinations
import scipy.stats
from scipy.optimize import shgo
from . import distributions
from ._common import ConfidenceInterval
from ._continuous_distns import chi2, norm
from scipy.special import gamma, kv, gammaln
from scipy.fft import ifft
from ._stats_pythran import _a_ij_Aij_Dij2
from ._stats_pythran import (
from ._axis_nan_policy import _axis_nan_policy_factory
from scipy.stats import _stats_py
def _pval_cvm_2samp_exact(s, m, n):
    """
    Compute the exact p-value of the Cramer-von Mises two-sample test
    for a given value s of the test statistic.
    m and n are the sizes of the samples.

    [1] Y. Xiao, A. Gordon, and A. Yakovlev, "A C++ Program for
        the Cramér-Von Mises Two-Sample Test", J. Stat. Soft.,
        vol. 17, no. 8, pp. 1-15, Dec. 2006.
    [2] T. W. Anderson "On the Distribution of the Two-Sample Cramer-von Mises
        Criterion," The Annals of Mathematical Statistics, Ann. Math. Statist.
        33(3), 1148-1159, (September, 1962)
    """
    lcm = np.lcm(m, n)
    a = lcm // m
    b = lcm // n
    mn = m * n
    zeta = lcm ** 2 * (m + n) * (6 * s - mn * (4 * mn - 1)) // (6 * mn ** 2)
    zeta_bound = lcm ** 2 * (m + n)
    combinations = comb(m + n, m)
    max_gs = max(zeta_bound, combinations)
    dtype = np.min_scalar_type(max_gs)
    gs = [np.array([[0], [1]], dtype=dtype)] + [np.empty((2, 0), dtype=dtype) for _ in range(m)]
    for u in range(n + 1):
        next_gs = []
        tmp = np.empty((2, 0), dtype=dtype)
        for v, g in enumerate(gs):
            vi, i0, i1 = np.intersect1d(tmp[0], g[0], return_indices=True)
            tmp = np.concatenate([np.stack([vi, tmp[1, i0] + g[1, i1]]), np.delete(tmp, i0, 1), np.delete(g, i1, 1)], 1)
            res = (a * v - b * u) ** 2
            tmp[0] += res.astype(dtype)
            next_gs.append(tmp)
        gs = next_gs
    value, freq = gs[m]
    return np.float64(np.sum(freq[value >= zeta]) / combinations)