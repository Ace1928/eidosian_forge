import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale
def test_tpvalues(self):
    params = self.res1.params
    tvalues = params / self.res1.bse
    pvalues = stats.norm.sf(np.abs(tvalues)) * 2
    half_width = stats.norm.isf(0.025) * self.res1.bse
    conf_int = np.column_stack((params - half_width, params + half_width))
    assert_almost_equal(self.res1.tvalues, tvalues)
    assert_almost_equal(self.res1.pvalues, pvalues)
    assert_almost_equal(self.res1.conf_int(), conf_int)