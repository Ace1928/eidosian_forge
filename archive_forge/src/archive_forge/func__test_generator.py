from warnings import warn as _warn
from math import log as _log, exp as _exp, pi as _pi, e as _e, ceil as _ceil
from math import sqrt as _sqrt, acos as _acos, cos as _cos, sin as _sin
from math import tau as TWOPI, floor as _floor, isfinite as _isfinite
from os import urandom as _urandom
from _collections_abc import Set as _Set, Sequence as _Sequence
from operator import index as _index
from itertools import accumulate as _accumulate, repeat as _repeat
from bisect import bisect as _bisect
import os as _os
import _random
def _test_generator(n, func, args):
    from statistics import stdev, fmean as mean
    from time import perf_counter
    t0 = perf_counter()
    data = [func(*args) for i in _repeat(None, n)]
    t1 = perf_counter()
    xbar = mean(data)
    sigma = stdev(data, xbar)
    low = min(data)
    high = max(data)
    print(f'{t1 - t0:.3f} sec, {n} times {func.__name__}')
    print('avg %g, stddev %g, min %g, max %g\n' % (xbar, sigma, low, high))