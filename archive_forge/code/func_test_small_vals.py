import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_raises,
import numpy.testing as npt
from scipy.special import gamma, factorial, factorial2
import scipy.stats as stats
from statsmodels.distributions.edgeworth import (_faa_di_bruno_partitions,
def test_small_vals(self):
    for n in range(1, 5):
        for ks in _faa_di_bruno_partitions(n):
            lhs = sum((m * k for m, k in ks))
            assert_equal(lhs, n)