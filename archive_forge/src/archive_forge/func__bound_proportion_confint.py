from statsmodels.compat.python import lzip
from typing import Callable
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
def _bound_proportion_confint(func: Callable[[float], float], qi: float, lower: bool=True) -> float:
    """
    Try hard to find a bound different from eps/1 - eps in proportion_confint

    Parameters
    ----------
    func : callable
        Callable function to use as the objective of the search
    qi : float
        The empirical success rate
    lower : bool
        Whether to fund a lower bound for the left side of the CI

    Returns
    -------
    float
        The coarse bound
    """
    default = FLOAT_INFO.eps if lower else 1.0 - FLOAT_INFO.eps

    def step(v):
        return v / 8 if lower else v + (1.0 - v) / 8
    x = step(qi)
    w = func(x)
    cnt = 1
    while w > 0 and cnt < 10:
        x = step(x)
        w = func(x)
        cnt += 1
    return x if cnt < 10 else default