import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from statsmodels.stats.multitest import (multipletests, fdrcorrection,
from statsmodels.stats.multicomp import tukeyhsd
from scipy.stats.distributions import norm
import scipy
from packaging import version
def test_null_distribution():
    grid = np.linspace(0.001, 0.999, 1000)
    z0 = norm.ppf(grid)
    z1 = np.linspace(3, 4, 20)
    zs = np.concatenate((z0, z1))
    emp_null = NullDistribution(zs, estimate_null_proportion=True)
    assert_allclose(emp_null.mean, 0, atol=1e-05, rtol=1e-05)
    assert_allclose(emp_null.sd, 1, atol=1e-05, rtol=0.01)
    assert_allclose(emp_null.null_proportion, 0.98, atol=1e-05, rtol=0.01)
    assert_allclose(emp_null.pdf(np.r_[-1, 0, 1]), norm.pdf(np.r_[-1, 0, 1], loc=emp_null.mean, scale=emp_null.sd), rtol=1e-13)