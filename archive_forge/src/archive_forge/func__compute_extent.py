from collections import namedtuple
import itertools
import functools
import operator
import ctypes
import numpy as np
from numba import _helperlib
from numba.core import config
def _compute_extent(self):
    firstidx = [0] * self.ndim
    lastidx = [s - 1 for s in self.shape]
    start = compute_index(firstidx, self.dims)
    stop = compute_index(lastidx, self.dims) + self.itemsize
    stop = max(stop, start)
    return Extent(start, stop)