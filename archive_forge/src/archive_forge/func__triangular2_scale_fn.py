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
def _triangular2_scale_fn(x):
    return 1 / 2.0 ** (x - 1)