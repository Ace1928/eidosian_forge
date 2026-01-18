import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises)
from statsmodels.base.transform import (BoxCox)
from statsmodels.datasets import macrodata
def test_valid_guerrero(self):
    lmbda = self.bc._est_lambda(self.x, method='guerrero', window_length=4)
    assert_almost_equal(lmbda, 0.507624, 4)
    lmbda = self.bc._est_lambda(self.x, method='guerrero', window_length=2)
    assert_almost_equal(lmbda, 0.513893, 4)