from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas.compat._optional import import_optional_dependency
class EWMMeanState:

    def __init__(self, com, adjust, ignore_na, axis, shape) -> None:
        alpha = 1.0 / (1.0 + com)
        self.axis = axis
        self.shape = shape
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.new_wt = 1.0 if adjust else alpha
        self.old_wt_factor = 1.0 - alpha
        self.old_wt = np.ones(self.shape[self.axis - 1])
        self.last_ewm = None

    def run_ewm(self, weighted_avg, deltas, min_periods, ewm_func):
        result, old_wt = ewm_func(weighted_avg, deltas, min_periods, self.old_wt_factor, self.new_wt, self.old_wt, self.adjust, self.ignore_na)
        self.old_wt = old_wt
        self.last_ewm = result[-1]
        return result

    def reset(self) -> None:
        self.old_wt = np.ones(self.shape[self.axis - 1])
        self.last_ewm = None