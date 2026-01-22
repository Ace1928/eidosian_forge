import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math
import itertools
import warnings
from collections import namedtuple
from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
from ._stats_mstats_common import (
def _quantiles1D(data, m, p):
    x = np.sort(data.compressed())
    n = len(x)
    if n == 0:
        return ma.array(np.empty(len(p), dtype=float), mask=True)
    elif n == 1:
        return ma.array(np.resize(x, p.shape), mask=nomask)
    aleph = n * p + m
    k = np.floor(aleph.clip(1, n - 1)).astype(int)
    gamma = (aleph - k).clip(0, 1)
    return (1.0 - gamma) * x[(k - 1).tolist()] + gamma * x[k.tolist()]