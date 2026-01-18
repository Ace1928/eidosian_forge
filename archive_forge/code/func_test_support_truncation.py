import pytest
import warnings
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
from copy import deepcopy
from scipy.stats.sampling import FastGeneratorInversion
from scipy import stats
@pytest.mark.parametrize('distname, args', [('beta', (3.5, 2.5)), ('norm', ())])
def test_support_truncation(distname, args):
    dist = getattr(stats, distname)(*args)
    rng = FastGeneratorInversion(dist, domain=(0.5, 0.7))
    assert_array_equal(rng.support(), (0.5, 0.7))
    rng.loc = 1
    rng.scale = 2
    assert_array_equal(rng.support(), (1 + 2 * 0.5, 1 + 2 * 0.7))