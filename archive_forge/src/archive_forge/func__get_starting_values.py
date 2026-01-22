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
def _get_starting_values(self, params, start_params, use_brute, sel, hw_args, bounds, alpha, func):
    if start_params is None and use_brute and np.any(sel[:3]):
        m = self.seasonal_periods
        sv_sel = np.array([False] * (6 + m))
        sv_sel[:3] = True
        sv_sel &= sel
        hw_args.xi = sv_sel.astype(np.int64)
        hw_args.transform = False
        points = self._setup_brute(sv_sel, bounds, alpha)
        opt = opt_wrapper(func)
        best_val = np.inf
        best_params = points[0]
        for point in points:
            val = opt(point, hw_args)
            if val < best_val:
                best_params = point
                best_val = val
        params[sv_sel] = best_params
    elif start_params is not None:
        if len(start_params) != sel.sum():
            msg = 'start_params must have {0} values but has {1}.'
            nxi, nsp = (len(sel), len(start_params))
            raise ValueError(msg.format(nxi, nsp))
        params[sel] = start_params
    return params