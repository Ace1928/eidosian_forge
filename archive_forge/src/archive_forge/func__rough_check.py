import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def _rough_check(a, b, compare_assert=partial(assert_allclose, atol=1e-05), key=lambda x: x, w=None):
    check_a = key(a)
    check_b = key(b)
    try:
        if np.array(check_a != check_b).any():
            compare_assert(check_a, check_b)
    except AttributeError:
        compare_assert(check_a, check_b)
    except (TypeError, ValueError):
        for a_i, b_i in zip(check_a, check_b):
            _rough_check(a_i, b_i, compare_assert=compare_assert)