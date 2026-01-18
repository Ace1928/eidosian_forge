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
@pytest.mark.parametrize('dist,shape_args', distcont)
def test_rvs_broadcast(dist, shape_args):
    if dist in ['gausshyper', 'studentized_range']:
        pytest.skip('too slow')
    if dist in ['rel_breitwigner'] and _IS_32BIT:
        pytest.skip('fails on Linux 32-bit')
    shape_only = dist in ['argus', 'betaprime', 'dgamma', 'dweibull', 'exponnorm', 'genhyperbolic', 'geninvgauss', 'levy_stable', 'nct', 'norminvgauss', 'rice', 'skewnorm', 'semicircular', 'gennorm', 'loggamma']
    distfunc = getattr(stats, dist)
    loc = np.zeros(2)
    scale = np.ones((3, 1))
    nargs = distfunc.numargs
    allargs = []
    bshape = [3, 2]
    for k in range(nargs):
        shp = (k + 4,) + (1,) * (k + 2)
        allargs.append(shape_args[k] * np.ones(shp))
        bshape.insert(0, k + 4)
    allargs.extend([loc, scale])
    check_rvs_broadcast(distfunc, dist, allargs, bshape, shape_only, 'd')