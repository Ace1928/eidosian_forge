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
def test_sokalmichener_with_weight():
    ntf = 0 * 1 + 0 * 0.2
    nft = 0 * 1 + 1 * 0.2
    ntt = 1 * 1 + 0 * 0.2
    nff = 0 * 1 + 0 * 0.2
    expected = 2 * (nft + ntf) / (ntt + nff + 2 * (nft + ntf))
    assert_almost_equal(expected, 0.2857143)
    actual = sokalmichener([1, 0], [1, 1], w=[1, 0.2])
    assert_almost_equal(expected, actual)
    a1 = [False, False, True, True, True, False, False, True, True, True, True, True, True, False, True, False, False, False, True, True]
    a2 = [True, True, True, False, False, True, True, True, False, True, True, True, True, True, False, False, False, True, True, True]
    for w in [0.05, 0.1, 1.0, 20.0]:
        assert_almost_equal(sokalmichener(a2, a1, [w]), 0.6666666666666666)