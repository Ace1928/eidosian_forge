import pytest
import warnings
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
from copy import deepcopy
from scipy.stats.sampling import FastGeneratorInversion
from scipy import stats
@pytest.mark.parametrize('domain, x', [(None, 0.5), ((0, 1), 0.5), ((0, 1), 1.5)])
def test_scalar_inputs(domain, x):
    """ pdf, cdf etc should map scalar values to scalars. check with and
    w/o domain since domain impacts pdf, cdf etc
    Take x inside and outside of domain """
    rng = FastGeneratorInversion(stats.norm(), domain=domain)
    assert np.isscalar(rng._cdf(x))
    assert np.isscalar(rng._ppf(0.5))