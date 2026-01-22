from statsmodels.compat.pandas import deprecate_kwarg
import contextlib
from typing import Any
from collections.abc import Hashable, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, least_squares, minimize
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from statsmodels.tools.validation import (
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.exponential_smoothing.ets import (
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import HoltWintersArgs
from statsmodels.tsa.holtwinters._smoothers import (
from statsmodels.tsa.holtwinters.results import (
from statsmodels.tsa.tsatools import freq_to_period
def _enforce_bounds(self, p, sel, lb, ub):
    initial_p = p[sel]
    loc = initial_p <= lb
    upper = ub[loc].copy()
    upper[~np.isfinite(upper)] = 100.0
    eps = 0.0001
    initial_p[loc] = lb[loc] + eps * (upper - lb[loc])
    loc = initial_p >= ub
    lower = lb[loc].copy()
    lower[~np.isfinite(lower)] = -100.0
    eps = 0.0001
    initial_p[loc] = ub[loc] - eps * (ub[loc] - lower)
    return initial_p