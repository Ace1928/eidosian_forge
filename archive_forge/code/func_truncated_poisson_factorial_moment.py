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
def truncated_poisson_factorial_moment(interval, r, p):
    """
            Compute mu_r, the r-th factorial moment of a poisson random
            variable of parameter `p` truncated to `interval = (b, a)`.
            """
    b, a = interval
    return p ** r * (1 - (poisson_interval((a - r + 1, a), p) - poisson_interval((b - r, b - 1), p)) / poisson_interval((b, a), p))