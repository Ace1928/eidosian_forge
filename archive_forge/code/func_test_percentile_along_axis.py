import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def test_percentile_along_axis():
    shape = (10, 20)
    np.random.seed(0)
    x = np.random.rand(*shape)
    q = np.random.rand(*shape[:-1]) * 100
    y = _resampling._percentile_along_axis(x, q)
    for i in range(shape[0]):
        res = y[i]
        expected = np.percentile(x[i], q[i], axis=-1)
        assert_allclose(res, expected, 1e-15)