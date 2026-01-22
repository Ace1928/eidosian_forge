from __future__ import print_function, absolute_import, division
import pytest
from .test_core import f, _test_powell
def _powell_by_name(x, params, backend=None):
    A, exp = (params['A'], backend.exp)
    u, v = (x['u'], x['v'])
    return (A * u * v - 1, exp(-u) + exp(-v) - (1 + A ** (-1)))