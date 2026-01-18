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
def ttest_1samp(a, popmean, axis=0, alternative='two-sided'):
    """
    Calculates the T-test for the mean of ONE group of scores.

    Parameters
    ----------
    a : array_like
        sample observation
    popmean : float or array_like
        expected value in null hypothesis, if array_like than it must have the
        same shape as `a` excluding the axis dimension
    axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole
        array `a`.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the mean of the underlying distribution of the sample
          is different than the given population mean (`popmean`)
        * 'less': the mean of the underlying distribution of the sample is
          less than the given population mean (`popmean`)
        * 'greater': the mean of the underlying distribution of the sample is
          greater than the given population mean (`popmean`)

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : float or array
        t-statistic
    pvalue : float or array
        The p-value

    Notes
    -----
    For more details on `ttest_1samp`, see `scipy.stats.ttest_1samp`.

    """
    a, axis = _chk_asarray(a, axis)
    if a.size == 0:
        return (np.nan, np.nan)
    x = a.mean(axis=axis)
    v = a.var(axis=axis, ddof=1)
    n = a.count(axis=axis)
    df = ma.asanyarray(n - 1.0)
    svar = (n - 1.0) * v / df
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (x - popmean) / ma.sqrt(svar / n)
    t, prob = scipy.stats._stats_py._ttest_finish(df, t, alternative)
    return Ttest_1sampResult(t, prob)