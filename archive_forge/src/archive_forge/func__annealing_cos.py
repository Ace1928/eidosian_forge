import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
@staticmethod
def _annealing_cos(start, end, pct):
    """Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out