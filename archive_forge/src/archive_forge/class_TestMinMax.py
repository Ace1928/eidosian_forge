import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
class TestMinMax:

    def test_minmax_blocked(self):
        for dt, sz in [(np.float32, 15), (np.float64, 7)]:
            for out, inp, msg in _gen_alignment_data(dtype=dt, type='unary', max_size=sz):
                for i in range(inp.size):
                    inp[:] = np.arange(inp.size, dtype=dt)
                    inp[i] = np.nan
                    emsg = lambda: '%r\n%s' % (inp, msg)
                    with suppress_warnings() as sup:
                        sup.filter(RuntimeWarning, 'invalid value encountered in reduce')
                        assert_(np.isnan(inp.max()), msg=emsg)
                        assert_(np.isnan(inp.min()), msg=emsg)
                    inp[i] = 10000000000.0
                    assert_equal(inp.max(), 10000000000.0, err_msg=msg)
                    inp[i] = -10000000000.0
                    assert_equal(inp.min(), -10000000000.0, err_msg=msg)

    def test_lower_align(self):
        d = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        assert_equal(d.max(), d[0])
        assert_equal(d.min(), d[0])

    def test_reduce_reorder(self):
        for n in (2, 4, 8, 16, 32):
            for dt in (np.float32, np.float16, np.complex64):
                for r in np.diagflat(np.array([np.nan] * n, dtype=dt)):
                    assert_equal(np.min(r), np.nan)

    def test_minimize_no_warns(self):
        a = np.minimum(np.nan, 1)
        assert_equal(a, np.nan)