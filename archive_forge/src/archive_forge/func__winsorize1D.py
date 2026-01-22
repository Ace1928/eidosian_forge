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
def _winsorize1D(a, low_limit, up_limit, low_include, up_include, contains_nan, nan_policy):
    n = a.count()
    idx = a.argsort()
    if contains_nan:
        nan_count = np.count_nonzero(np.isnan(a))
    if low_limit:
        if low_include:
            lowidx = int(low_limit * n)
        else:
            lowidx = np.round(low_limit * n).astype(int)
        if contains_nan and nan_policy == 'omit':
            lowidx = min(lowidx, n - nan_count - 1)
        a[idx[:lowidx]] = a[idx[lowidx]]
    if up_limit is not None:
        if up_include:
            upidx = n - int(n * up_limit)
        else:
            upidx = n - np.round(n * up_limit).astype(int)
        if contains_nan and nan_policy == 'omit':
            a[idx[upidx:-nan_count]] = a[idx[upidx - 1]]
        else:
            a[idx[upidx:]] = a[idx[upidx - 1]]
    return a