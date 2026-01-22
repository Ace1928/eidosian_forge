import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
def _init_is_better(self, mode, threshold, threshold_mode):
    if mode not in {'min', 'max'}:
        raise ValueError('mode ' + mode + ' is unknown!')
    if threshold_mode not in {'rel', 'abs'}:
        raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')
    if mode == 'min':
        self.mode_worse = inf
    else:
        self.mode_worse = -inf
    self.mode = mode
    self.threshold = threshold
    self.threshold_mode = threshold_mode