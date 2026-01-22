import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_raises,
import numpy.testing as npt
from scipy.special import gamma, factorial, factorial2
import scipy.stats as stats
from statsmodels.distributions.edgeworth import (_faa_di_bruno_partitions,
def _chi2_cumulant(n, df):
    assert n > 0
    return 2 ** (n - 1) * factorial(n - 1) * df