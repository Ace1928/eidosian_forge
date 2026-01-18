import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
from numpy.testing import (assert_array_less, assert_almost_equal,
def test_use_t_summary(self):
    orig_val = self.res1.use_t
    self.res1.use_t = True
    summ = self.res1.summary()
    assert 'P>|t|' in str(summ)
    self.res1.use_t = orig_val