import pytest
import warnings
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
from copy import deepcopy
from scipy.stats.sampling import FastGeneratorInversion
from scipy import stats
def test_setting_loc_scale():
    rng = FastGeneratorInversion(stats.norm(), random_state=765765864)
    r1 = rng.rvs(size=1000)
    rng.loc = 3.0
    rng.scale = 2.5
    r2 = rng.rvs(1000)
    assert stats.cramervonmises_2samp(r1, (r2 - 3) / 2.5).pvalue > 0.05
    rng.loc = 0
    rng.scale = 1
    r2 = rng.rvs(1000)
    assert stats.cramervonmises_2samp(r1, r2).pvalue > 0.05