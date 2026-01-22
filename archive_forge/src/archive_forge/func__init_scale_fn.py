import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
def _init_scale_fn(self):
    if self._scale_fn_custom is not None:
        return
    if self.mode == 'triangular':
        self._scale_fn_ref = self._triangular_scale_fn
        self.scale_mode = 'cycle'
    elif self.mode == 'triangular2':
        self._scale_fn_ref = self._triangular2_scale_fn
        self.scale_mode = 'cycle'
    elif self.mode == 'exp_range':
        self._scale_fn_ref = partial(self._exp_range_scale_fn, self.gamma)
        self.scale_mode = 'iterations'