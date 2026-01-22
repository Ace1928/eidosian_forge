import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
def _student_t_cdf(df, t, dps=None):
    if dps is None:
        dps = mpmath.mp.dps
    with mpmath.workdps(dps):
        df, t = (mpmath.mpf(df), mpmath.mpf(t))
        fac = mpmath.hyp2f1(0.5, 0.5 * (df + 1), 1.5, -t ** 2 / df)
        fac *= t * mpmath.gamma(0.5 * (df + 1))
        fac /= mpmath.sqrt(mpmath.pi * df) * mpmath.gamma(0.5 * df)
        return 0.5 + fac