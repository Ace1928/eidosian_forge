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
def _all_partitions(nx, ny):
    """
    Partition a set of indices into two fixed-length sets in all possible ways

    Partition a set of indices 0 ... nx + ny - 1 into two sets of length nx and
    ny in all possible ways (ignoring order of elements).
    """
    z = np.arange(nx + ny)
    for c in combinations(z, nx):
        x = np.array(c)
        mask = np.ones(nx + ny, bool)
        mask[x] = False
        y = z[mask]
        yield (x, y)