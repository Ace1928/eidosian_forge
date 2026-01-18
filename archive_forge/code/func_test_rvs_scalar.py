import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
@pytest.mark.parametrize('distname,arg', cases_test_cont_basic())
def test_rvs_scalar(distname, arg):
    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'rv_histogram_instance'
    assert np.isscalar(distfn.rvs(*arg))
    assert np.isscalar(distfn.rvs(*arg, size=()))
    assert np.isscalar(distfn.rvs(*arg, size=None))