import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises)
from statsmodels.base.transform import (BoxCox)
from statsmodels.datasets import macrodata
def test_zero_lambda(self):
    y_transform_zero_lambda, lmbda = self.bc.transform_boxcox(self.x, 0.0)
    assert_equal(lmbda, 0.0)
    assert_almost_equal(y_transform_zero_lambda, np.log(self.x), 5)