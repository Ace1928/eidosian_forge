import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
def _binomial_cdf(k, n, p):
    k, n, p = (mpmath.mpf(k), mpmath.mpf(n), mpmath.mpf(p))
    if k <= 0:
        return mpmath.mpf(0)
    elif k >= n:
        return mpmath.mpf(1)
    onemp = mpmath.fsub(1, p, exact=True)
    return mpmath.betainc(n - k, k + 1, x2=onemp, regularized=True)