import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises)
from statsmodels.base.transform import (BoxCox)
from statsmodels.datasets import macrodata
def test_unclear_scale_parameter(self):
    assert_raises(ValueError, self.bc._est_lambda, self.x, scale='test')
    self.bc._est_lambda(self.x, scale='mad')
    self.bc._est_lambda(self.x, scale='MAD')
    self.bc._est_lambda(self.x, scale='sd')
    self.bc._est_lambda(self.x, scale='SD')