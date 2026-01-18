import os
import sys
import time
from itertools import zip_longest
import numpy as np
from numpy.testing import assert_
import pytest
from scipy.special._testutils import assert_func_equal
def trace_args(func):

    def tofloat(x):
        if isinstance(x, mpmath.mpc):
            return complex(x)
        else:
            return float(x)

    def wrap(*a, **kw):
        sys.stderr.write(f'{tuple(map(tofloat, a))!r}: ')
        sys.stderr.flush()
        try:
            r = func(*a, **kw)
            sys.stderr.write('-> %r' % r)
        finally:
            sys.stderr.write('\n')
            sys.stderr.flush()
        return r
    return wrap