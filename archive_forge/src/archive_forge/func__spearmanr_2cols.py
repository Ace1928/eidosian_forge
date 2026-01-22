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
def _spearmanr_2cols(x):
    x = ma.mask_rowcols(x, axis=0)
    x = x[~x.mask.any(axis=1), :]
    if not np.any(x.data):
        res = scipy.stats._stats_py.SignificanceResult(np.nan, np.nan)
        res.correlation = np.nan
        return res
    m = ma.getmask(x)
    n_obs = x.shape[0]
    dof = n_obs - 2 - int(m.sum(axis=0)[0])
    if dof < 0:
        raise ValueError('The input must have at least 3 entries!')
    x_ranked = rankdata(x, axis=0)
    rs = ma.corrcoef(x_ranked, rowvar=False).data
    with np.errstate(divide='ignore'):
        t = rs * np.sqrt((dof / ((rs + 1.0) * (1.0 - rs))).clip(0))
    t, prob = scipy.stats._stats_py._ttest_finish(dof, t, alternative)
    if rs.shape == (2, 2):
        res = scipy.stats._stats_py.SignificanceResult(rs[1, 0], prob[1, 0])
        res.correlation = rs[1, 0]
        return res
    else:
        res = scipy.stats._stats_py.SignificanceResult(rs, prob)
        res.correlation = rs
        return res