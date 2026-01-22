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
def _tau_b(A):
    """Calculate Kendall's tau-b and p-value from contingency table."""
    if A.shape[0] == 1 or A.shape[1] == 1:
        return (np.nan, np.nan)
    NA = A.sum()
    PA = _P(A)
    QA = _Q(A)
    Sri2 = (A.sum(axis=1) ** 2).sum()
    Scj2 = (A.sum(axis=0) ** 2).sum()
    denominator = (NA ** 2 - Sri2) * (NA ** 2 - Scj2)
    tau = (PA - QA) / denominator ** 0.5
    numerator = 4 * (_a_ij_Aij_Dij2(A) - (PA - QA) ** 2 / NA)
    s02_tau_b = numerator / denominator
    if s02_tau_b == 0:
        return (tau, 0)
    Z = tau / s02_tau_b ** 0.5
    p = 2 * norm.sf(abs(Z))
    return (tau, p)