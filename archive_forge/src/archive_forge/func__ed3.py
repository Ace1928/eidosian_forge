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
def _ed3(y):
    z = y ** 2 / 4
    c = np.exp(-z) / np.sqrt(np.pi)
    return c * (y / 2) ** (5 / 2) * (2 * kv(1 / 4, z) + 3 * kv(3 / 4, z) - kv(5 / 4, z))