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
def approximated_multinomial_interval(intervals):
    """
            Compute approximated probability for Multinomial(n, proportions)
            to be in `intervals` (Sison & Glaz's formula (1)).
            """
    return np.exp(np.sum(np.log([poisson_interval(interval, p) for interval, p in zip(intervals, counts)])) + np.log(edgeworth(intervals)) - np.log(stats.poisson._pmf(n, n)))