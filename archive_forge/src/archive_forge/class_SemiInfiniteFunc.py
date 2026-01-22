import sys
import copy
import heapq
import collections
import functools
import numpy as np
from scipy._lib._util import MapWrapper, _FunctionWrapper
class SemiInfiniteFunc:
    """
    Argument transform from (start, +-oo) to (0, 1)
    """

    def __init__(self, func, start, infty):
        self._func = func
        self._start = start
        self._sgn = -1 if infty < 0 else 1
        self._tmin = sys.float_info.min ** 0.5

    def get_t(self, x):
        z = self._sgn * (x - self._start) + 1
        if z == 0:
            return np.inf
        return 1 / z

    def __call__(self, t):
        if t < self._tmin:
            return 0.0
        else:
            x = self._start + self._sgn * (1 - t) / t
            f = self._func(x)
            return self._sgn * (f / t) / t